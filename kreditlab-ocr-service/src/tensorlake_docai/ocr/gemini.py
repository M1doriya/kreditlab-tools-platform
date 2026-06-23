# SPDX-License-Identifier: Apache-2.0
import time
import os
import json
import re
import asyncio
from typing import List, Dict, Optional, Union, Tuple, Any

from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.vlm.workflow_images import simple_page_creator_image
from tensorlake_docai.pipeline.api import PageFragmentType, Usage
from tensorlake_docai.models.layout_objects import PageLayoutElement, PageLayout, DocumentLayout
from tensorlake.applications import Retries, cls, function
from tensorlake.applications import RequestError as RequestException

from tensorlake_docai.postprocess.header_correction import correct_document_headers

from tensorlake_docai.pipeline.routing import route_after_ocr
from tensorlake_docai.pipeline.simple_page_creator import SimplePageCreator
from tensorlake_docai.extraction.form_extraction_utils import convert_form_json_to_markdown

GEMINI_MODEL = "gemini-3.1-flash-lite-preview"
PROVIDER_NAME = "GEMINI"

PROMPT = (
    "Extract the document content as a list of fragments in strict reading order. "
    "For each fragment, provide the 'fragment_type', 'text', and 'bbox'.\n"
    "Supported fragment_types:\n"
    "TEXT, TITLE, SECTION_HEADER, PAGE_HEADER, PAGE_FOOTER, PAGE_NUMBER, TABLE, FORM, FIGURE, SIGNATURE, FORMULA, PAGE_BREAK.\n\n"
    "Rules:\n"
    "1) All text should be extracted, no text that appears in the document image should be missed, but do not extract text that is not in the page image, you cannot interprete what is in the page image. If the fragment type is unclear, it will be considered as a fragment of type text.\n"
    "2) Fragments should be created in strict reading order.\n"
    "3) All fragments should have a bbox defined as [ymin, xmin, ymax, xmax] in 0-1000 scale.\n"
    "4) For TABLE, 'text' should be the HTML representation of the table, with no decorative or formatting tags are required, but table headers should be identified. "
    "5) For FIGURE, 'text' should be a concise summary (less than 200 words) describing the figure's content, purpose, and any key values if present.\n"
    "6) For FORM fragments, return a list of key-value pairs in JSON format.\n"
    "7) Do NOT include decorative separators or repeated punctuation/lines (e.g., '..........', '--------', '________', '=======', or spaced dot patterns). If such lines exist visually, omit them. Never repeat any single character more than three times in a row.\n"
    "8) Insert PAGE_BREAK between pages for multi-page documents.\n"
    "9) If a paragraphs spans two columns, they should be in two fragments, since otherwise the bounding box cannot be properly specified."
    "10) Be sure that no additional text added not present in the document, check for repetitions of hallucinated text. "
    "11) Pay special attention to non-English text in scanned documents, make sure to transcribe it properly and do not replace it with English or other characters."
)

RESPONSE_SCHEMA = {
    "type": "array",
    "items": {
        "type": "object",
        "properties": {
            "fragment_type": {
                "type": "string",
                "enum": [
                    "TITLE",
                    "SECTION_HEADER",
                    "PAGE_HEADER",
                    "PAGE_FOOTER",
                    "PAGE_NUMBER",
                    "TABLE",
                    "FORM",
                    "FIGURE",
                    "SIGNATURE",
                    "FORMULA",
                    "PAGE_BREAK",
                    "TEXT",
                ],
            },
            "text": {
                "type": "string",
                "description": "Content of the fragment. For TABLE types, this should be HTML using <table> tags. For all other types, this is plain text. For forms it will be JSON.",
            },
            "bbox": {
                "type": "array",
                "items": {"type": "integer"},
                "minItems": 4,
                "maxItems": 4,
            },
        },
        "required": ["fragment_type", "text", "bbox"],
    },
}

SYSTEM_INSTRUCTION = (
    "You are an expert OCR model. You are given a page image and you need to transcribe the text in the image into fragments sorted in human reading order. "
    "Be sure to write proper HTML and JSON when requested. "
    "Be aware of non-English texts to produce proper JSON. "
    "There is no document with prohibited material, no need to refuse to answer. "
    "Human reading order is the logical, sequential path a person’s eyes follow to consume content based on language direction (e.g., LTR in English, RTL in Arabic/Hebrew) and visual hierarchy. It ensures text and elements are processed chronologically and contextually as intended by the creator."
)

SECRETS = [
    "GEMINI_API_KEY",
    "OPENAI_API_KEY",
    "USE_AZURE_OPENAI",
    "AZURE_OPENAI_ENDPOINT",
    "AZURE_OPENAI_API_KEY",
    "AZURE_OPENAI_MODEL_DEPLOYMENT_NAME",
]

MEMORY_IN_GB = 8

# Tags and compiled regexes
TAG_NAMES = "TITLE|SECTION_HEADER|PAGE_HEADER|PAGE_FOOTER|PAGE_NUMBER|PARAGRAPH|TABLE|FIGURE|SIGNATURE|FORMULA|PAGE_BREAK"
TAG_LINE_RE = re.compile(rf"^\[({TAG_NAMES})\]\s*(.*)$", re.MULTILINE)
LEADING_TAG_RE = re.compile(rf"^\[(?:{TAG_NAMES})\]\s*")
HTML_TAG_RE = re.compile(r"<[^>]+>")
BR_TAG_RE = re.compile(r"(?i)<br\s*/?>")
THOUSAND_SPLIT_RE = re.compile(r"(?<=\d),(?:\s*\n\s*)(?=\d{3}\b)")
LETTER_BREAK_RE = re.compile(r"(?<=[A-Za-z])\s*\n\s*(?=[A-Za-z])")
MULTISPACE_RE = re.compile(r"[ \t]{2,}")
DOLLAR_BLOCK_RE = re.compile(r"(?<!\\)\$\$")
DOLLAR_INLINE_RE = re.compile(r"(?<!\\)\$")


def sanitize_text(
    text: str,
    *,
    strip_html: bool = False,
    strip_leading_tag: bool = False,
    replace_br: bool = False,
    normalize_lines: bool = True,
    collapse_spaces: bool = True,
    escape_dollar: bool = False,
    final_trim: bool = True,
) -> str:
    if not text:
        return text
    if strip_leading_tag:
        text = LEADING_TAG_RE.sub("", text)
    if replace_br:
        text = BR_TAG_RE.sub("\n", text)
    if strip_html:
        text = HTML_TAG_RE.sub(" ", text)
    if normalize_lines:
        text = THOUSAND_SPLIT_RE.sub(",", text)
        text = LETTER_BREAK_RE.sub(" ", text)
    if collapse_spaces:
        text = MULTISPACE_RE.sub(" ", text)
    if escape_dollar:
        text = DOLLAR_BLOCK_RE.sub("\\$\\$", text)
        text = DOLLAR_INLINE_RE.sub("\\$", text)
    return text.strip() if final_trim else text


def sanitize_html_cell(cell: str) -> str:
    return sanitize_text(
        cell,
        replace_br=True,
        strip_html=True,
        escape_dollar=True,
    )


@cls()
class FullPageGeminiTask:  # type: ignore
    """Gemini OCR provider for text extraction with semantic structure tags."""

    def __init__(self):
        self._simple_page_creator = SimplePageCreator(scale_factor=2.0, memory_gb=MEMORY_IN_GB)
        self._gemini_api_key: Optional[str] = None

    @function(
        image=simple_page_creator_image,
        timeout=30 * 60,
        cpu=2,
        memory=MEMORY_IN_GB,
        ephemeral_disk=2,
        secrets=SECRETS,
        retries=Retries(max_retries=2),
        min_containers=int(os.getenv("TENSORLAKE_MIN_CONTAINERS", "0")),
    )
    def run(self, parse_result: ParseResult) -> ParseResult:
        """Main entry point for OCR processing."""
        if parse_result.request.mime_type.startswith("text/"):
            return parse_result

        print(f"Running FullPageGeminiTask ({PROVIDER_NAME})")
        start_time = time.time()

        try:
            self._initialize_model()

            # Run async processing
            asyncio.run(self._run_async(parse_result))

        except RequestException:
            raise
        except Exception as e:
            import traceback

            print(f"{PROVIDER_NAME} OCR failed: {str(e)}")
            print(f"Stack trace:\n{traceback.format_exc()}")
            raise RequestException(message=str(e))

        # Optional: header correction
        if parse_result.request.xpage_header_detection:
            try:
                parse_result = correct_document_headers(
                    parse_result, api_key=os.getenv("OPENAI_API_KEY")
                )
            except RequestException:
                raise
            except Exception as e:
                print(f"[FGT] Header correction skipped: {e}")

        print(f"FullPageGeminiTask completed in {time.time() - start_time:.2f} seconds")
        return route_after_ocr(parse_result, log_prefix=f"FULL_PAGE_{PROVIDER_NAME}")

    async def _run_async(self, parse_result: ParseResult) -> None:
        """Async processing helper."""
        # Route to PDF or image processing based on mime type
        if parse_result.request.mime_type == "application/pdf":
            await self._process_pdf(parse_result)
        else:
            await self._process_images(parse_result)

    def _initialize_model(self) -> None:
        """Initialize the Gemini API client."""
        self._gemini_api_key = os.getenv("GEMINI_API_KEY")
        if not self._gemini_api_key:
            print("Gemini API key missing")
            raise RequestException(
                message="Gemini OCR service temporarily unavailable. Please contact Tensorlake support with the trace ID of the job."
            )

    async def _process_pdf(self, parse_result: ParseResult) -> None:
        """Process PDF directly using Gemini's native PDF support."""
        import io
        import fitz  # PyMuPDF

        start = time.time()
        file_bytes = parse_result.request.file_bytes
        pdf_doc = fitz.open(stream=io.BytesIO(file_bytes), filetype="pdf")
        total_pages = len(pdf_doc)

        # Get page dimensions (same across all pages typically)
        page_width = pdf_doc[0].rect.width if total_pages > 0 else 612
        page_height = pdf_doc[0].rect.height if total_pages > 0 else 792

        # Extract specific pages if requested (Gemini API doesn't support pages parameter)
        pages_to_parse = (
            sorted(set(parse_result.request.pages_to_parse))
            if parse_result.request.pages_to_parse
            else None
        )
        if pages_to_parse:
            print(f"Extracting pages {pages_to_parse} from {total_pages} page PDF")
            new_pdf = fitz.open()
            for orig_num in pages_to_parse:
                if 1 <= orig_num <= total_pages:
                    new_pdf.insert_pdf(pdf_doc, from_page=orig_num - 1, to_page=orig_num - 1)

            pdf_bytes = new_pdf.tobytes()
            page_mapping = {
                new_num: orig_num
                for new_num, orig_num in enumerate(pages_to_parse, start=1)
                if 1 <= orig_num <= total_pages
            }
            parsed_pages = new_pdf.page_count
            new_pdf.close()
        else:
            pdf_bytes = file_bytes
            page_mapping = None
            parsed_pages = total_pages

        pdf_doc.close()
        print(f"PDF prepared: {time.time() - start:.2f}s | {parsed_pages}/{total_pages} pages")

        # Initialize document layout
        self._init_document_layout(parse_result, parsed_pages, total_pages, 1.0)

        # Call Gemini API with PDF bytes
        start = time.time()
        pages_sections, input_tokens, output_tokens = await self._run_gemini_ocr(
            pdf_bytes, is_pdf=True
        )
        print(
            f"Gemini processed {parsed_pages} pages in {time.time() - start:.2f}s | tokens={input_tokens}+{output_tokens}"
        )

        # Create page layouts
        for page_num in range(1, parsed_pages + 1):
            original_page_num = page_mapping[page_num] if page_mapping else page_num
            sections = pages_sections.get(page_num, [])
            page_layout = self._build_page_layout(
                original_page_num, int(page_width), int(page_height), sections
            )
            parse_result.document_layout.pages.append(page_layout)

        # Update token usage
        self._update_token_usage(parse_result, input_tokens, output_tokens)

    async def _process_images(self, parse_result: ParseResult) -> None:
        """Process images using SimplePageCreator and per-page Gemini calls."""
        start = time.time()

        # Convert to images
        document_pages = self._simple_page_creator.get_images(parse_result)
        print(f"Images prepared: {time.time() - start:.2f}s | {document_pages.total_pages} pages")

        # Initialize document layout
        self._init_document_layout(
            parse_result,
            document_pages.total_pages,
            document_pages.total_pages,
            document_pages.scale_factor,
        )

        # Process each page with Gemini using asyncio
        async def process_page(page_number: int, page_image: any):
            start = time.time()
            sections, input_tokens, output_tokens = await self._run_gemini_ocr(page_image)
            return (
                page_number,
                sections,
                input_tokens,
                output_tokens,
                page_image,
                time.time() - start,
            )

        # Create tasks for concurrent processing
        tasks = [
            process_page(page_num, page_img)
            for page_num, page_img in document_pages.page_images.items()
        ]

        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)

        # Sort by page number
        results = sorted(results, key=lambda r: r[0])

        # Create page layouts and accumulate tokens
        total_input_tokens = 0
        total_output_tokens = 0
        for page_num, sections, input_tokens, output_tokens, page_image, duration in results:
            total_input_tokens += input_tokens
            total_output_tokens += output_tokens
            page_layout = self._build_page_layout(
                page_num, page_image.size[0], page_image.size[1], sections
            )
            parse_result.document_layout.pages.append(page_layout)
            print(
                f"Page {page_num} processed: {duration:.2f}s | tokens={input_tokens}+{output_tokens}"
            )

        # Update token usage
        self._update_token_usage(parse_result, total_input_tokens, total_output_tokens)

    def _init_document_layout(
        self, parse_result: ParseResult, pages_parsed: int, total_pages: int, scale_factor: float
    ) -> None:
        """Initialize document layout and usage."""
        parse_result.usage = Usage(pages_parsed=pages_parsed)

        if not parse_result.document_layout:
            parse_result.document_layout = DocumentLayout(
                pages=[], scale_factor=scale_factor, total_pages=total_pages
            )
        else:
            parse_result.document_layout.scale_factor = scale_factor
            parse_result.document_layout.total_pages = total_pages

    def _update_token_usage(
        self, parse_result: ParseResult, input_tokens: int, output_tokens: int
    ) -> None:
        """Update token usage in parse result."""
        if parse_result.usage:
            parse_result.usage.ocr_input_tokens_used = (
                parse_result.usage.ocr_input_tokens_used or 0
            ) + input_tokens
            parse_result.usage.ocr_output_tokens_used = (
                parse_result.usage.ocr_output_tokens_used or 0
            ) + output_tokens

    async def _run_gemini_ocr(
        self, content_input: Union[bytes, Any], is_pdf: bool = False
    ) -> Union[Tuple[Dict[int, List[Dict]], int, int], Tuple[List[Dict], int, int]]:
        """Call Gemini API to extract text with semantic tags from PDF or image."""
        from tensorlake_docai.providers.model_provider_utils import _make_gemini_call

        # Prepare parameters for shared Gemini call
        if is_pdf:
            pdf_bytes = content_input
            images = []
        else:
            pdf_bytes = None
            images = [content_input]

        # Convert RESPONSE_SCHEMA to JSON string format
        json_schema = json.dumps(RESPONSE_SCHEMA)

        # OCR-specific config overrides for more deterministic output
        config_overrides = {
            "temperature": 0.0,
            "response_mime_type": "application/json",
            "max_output_tokens": 64000,
            "top_p": 0,
            "top_k": 1,
        }

        # Call shared streaming Gemini function
        response_text, input_tokens, output_tokens = await _make_gemini_call(
            user_prompt=PROMPT,
            images=images,
            page_image=None,
            json_schema=json_schema,
            job_type="ocr",  # Custom job type for OCR
            timeout=None,
            pdf_bytes=pdf_bytes,
            model_name=GEMINI_MODEL,
            system_instruction=SYSTEM_INSTRUCTION,
            config_overrides=config_overrides,
        )

        # Parse JSON response and extract content
        content_text = self._parse_response(response_text)

        # Return different formats based on input type
        if is_pdf:
            pages_sections = self._extract_pages_sections(content_text)
            return pages_sections, input_tokens, output_tokens
        else:
            sections = self._extract_sections(content_text)
            return sections, input_tokens, output_tokens

    def _parse_response(self, response_text: str) -> List[Dict]:
        """Parse JSON response."""
        try:
            return json.loads(response_text)
        except Exception:
            return []

    def _extract_pages_sections(self, content: List[Dict]) -> Dict[int, List[Dict[str, Any]]]:
        """Extract sections from multi-page content, splitting by PAGE_BREAK markers."""
        if not content:
            return {1: []}

        pages = {}
        current_page_num = 1
        current_page_content = []

        for fragment in content:
            if fragment.get("fragment_type") == "PAGE_BREAK":
                pages[current_page_num] = self._extract_sections(current_page_content)
                current_page_num += 1
                current_page_content = []
            else:
                current_page_content.append(fragment)

        if current_page_content:
            pages[current_page_num] = self._extract_sections(current_page_content)

        return pages

    def _extract_sections(self, content: List[Dict]) -> List[Dict[str, Any]]:
        """Extract tagged sections from content list."""
        sections = []
        for fragment in content:
            sections.append(
                {
                    "type": fragment.get("fragment_type", "PARAGRAPH").lower(),
                    "content": fragment.get("text", ""),
                    "bbox": fragment.get("bbox", [0, 0, 0, 0]),
                }
            )
        return sections

    def _build_page_layout(
        self, page_number: int, width: int, height: int, sections: List[Dict]
    ) -> PageLayout:
        """Build page layout with given dimensions and sections."""
        elements = [
            self._create_element(section, idx + 1, width, height)
            for idx, section in enumerate(sections)
        ]
        return PageLayout(
            page_number=page_number,
            elements=elements,
            shape=(height, width),
            page_dimensions={"width": width, "height": height},
        )

    def _create_element(
        self, section: Dict[str, Any], reading_order: int, page_width: int, page_height: int
    ) -> PageLayoutElement:
        """Create a page element from a section."""
        section_type = section["type"]
        content = section["content"]
        bbox = section.get("bbox", [0, 0, 0, 0])

        # Gemini bbox is [ymin, xmin, ymax, xmax] in 0-1000 scale
        ymin, xmin, ymax, xmax = bbox

        x1 = int(xmin * page_width / 1000)
        y1 = int(ymin * page_height / 1000)
        x2 = int(xmax * page_width / 1000)
        y2 = int(ymax * page_height / 1000)

        # Ensure coordinates are ordered correctly
        if x1 > x2:
            x1, x2 = x2, x1
        if y1 > y2:
            y1, y2 = y2, y1

        fragment_type = self._get_fragment_type(section_type)

        if section_type == "table":
            html_content = self._ensure_html_table(content)
            markdown_content = self._convert_html_to_markdown_table(html_content)
        elif section_type == "form":
            html_content = None
            markdown_content = convert_form_json_to_markdown(content)
        else:
            html_content = None
            markdown_content = sanitize_text(
                content or "",
                strip_leading_tag=True,
                replace_br=True,
                normalize_lines=True,
                collapse_spaces=True,
                escape_dollar=True,
                final_trim=True,
            )

        return PageLayoutElement(
            bbox=(x1, y1, x2, y2),
            fragment_type=fragment_type,
            score=1.0,
            reading_order=reading_order,
            ocr_text=sanitize_text(
                content,
                strip_html=True,
                strip_leading_tag=True,
                normalize_lines=True,
                collapse_spaces=True,
                escape_dollar=False,
                final_trim=True,
            ),
            markdown=markdown_content,
            html=html_content,
        )

    def _get_fragment_type(self, section_type: str) -> PageFragmentType:
        """Map section type to PageFragmentType."""
        mapping = {
            "title": PageFragmentType.TITLE,
            "form": PageFragmentType.FORM,
            "section_header": PageFragmentType.SECTION_HEADER,
            "page_header": PageFragmentType.PAGE_HEADER,
            "page_footer": PageFragmentType.PAGE_FOOTER,
            "page_number": PageFragmentType.PAGE_NUMBER,
            "paragraph": PageFragmentType.TEXT,
            "table": PageFragmentType.TABLE,
            "figure": PageFragmentType.FIGURE,
            "signature": PageFragmentType.SIGNATURE,
            "formula": PageFragmentType.FORMULA,
        }
        return mapping.get(section_type, PageFragmentType.TEXT)

    def _ensure_html_table(self, content: str) -> str:
        """Ensure content is a valid HTML table."""
        content = content.strip()

        # If already has table tags, return as-is
        if content.startswith("<table") and content.endswith("</table>"):
            return content

        # If it's a Markdown table, convert to HTML
        if "|" in content and self._is_markdown_table(content):
            return self._convert_markdown_to_html_table(content)

        # Otherwise wrap in table tags
        return f"<table>{content}</table>"

    def _is_markdown_table(self, text: str) -> bool:
        """Check if text is a Markdown table."""
        lines = text.strip().split("\n")
        if len(lines) < 2:
            return False

        # Check for separator line with dashes
        for line in lines:
            if re.match(r"^[\|\s\-:]+$", line):
                return True
        return False

    def _convert_markdown_to_html_table(self, markdown: str) -> str:
        """Convert Markdown table to HTML."""
        lines = [line.strip() for line in markdown.strip().split("\n")]
        if len(lines) < 2:
            return markdown

        html_parts = ["<table>"]

        # Process header
        header_cells = [cell.strip() for cell in lines[0].strip("|").split("|")]
        html_parts.append("<thead><tr>")
        for cell in header_cells:
            html_parts.append(f"<th>{cell}</th>")
        html_parts.append("</tr></thead>")

        # Process body (skip separator line)
        html_parts.append("<tbody>")
        for line in lines[2:]:
            if line.strip():
                cells = [cell.strip() for cell in line.strip("|").split("|")]
                html_parts.append("<tr>")
                for cell in cells:
                    html_parts.append(f"<td>{cell}</td>")
                html_parts.append("</tr>")
        html_parts.append("</tbody>")
        html_parts.append("</table>")

        return "".join(html_parts)

    def _convert_html_to_markdown_table(self, html: str) -> str:
        """Convert HTML table to Markdown."""
        # Extract cells from HTML (support multiline tags)
        header_cells = re.findall(r"<th[^>]*>([\s\S]*?)</th>", html, re.IGNORECASE)
        tbody_match = re.search(r"<tbody[^>]*>([\s\S]*?)</tbody>", html, re.IGNORECASE)
        tbody_html = tbody_match.group(1) if tbody_match else html
        body_rows = re.findall(r"<tr[^>]*>([\s\S]*?)</tr>", tbody_html, re.IGNORECASE)

        if not header_cells and not body_rows:
            return html

        markdown_lines = []

        # Build header
        if header_cells:
            sanitized_headers = []
            for cell in header_cells:
                sanitized_headers.append(sanitize_html_cell(cell).replace("\n", " ").strip())
            markdown_lines.append("| " + " | ".join(sanitized_headers) + " |")
            markdown_lines.append("| " + " | ".join(["---"] * len(header_cells)) + " |")

        # Build body
        for row_html in body_rows:
            cells = re.findall(r"<td[^>]*>([\s\S]*?)</td>", row_html, re.IGNORECASE)
            if cells:
                sanitized_cells = []
                for cell in cells:
                    sanitized_cells.append(sanitize_html_cell(cell).replace("\n", " ").strip())
                markdown_lines.append("| " + " | ".join(sanitized_cells) + " |")

        return "\n".join(markdown_lines)
