# SPDX-License-Identifier: Apache-2.0
import asyncio
from collections import Counter
from html.parser import HTMLParser
from typing import Any
from PIL import Image
import inspect

from tensorlake_docai.prompts.prompts import get_table_correction_prompt_messages

import re
import json
from tensorlake_docai.providers.model_provider_utils import run_clients, _make_gemini_call
from tensorlake.applications import RequestError as RequestException

try:
    import pytesseract

    HAS_OCR = True
except ImportError:
    HAS_OCR = False

HAS_OCR = False


class TableGridParser(HTMLParser):
    """Parses HTML table into a grid to validate dimensions."""

    def __init__(self):
        super().__init__()
        self.rows = []
        self.current_row = []
        self.in_cell = False
        self.current_colspan = 1
        self.current_rowspan = 1
        self.cell_content = []

    def handle_starttag(self, tag, attrs):
        if tag == "tr":
            self.current_row = []
        elif tag in ("td", "th"):
            self.in_cell = True
            self.current_colspan = 1
            self.current_rowspan = 1
            for k, v in attrs:
                if k == "colspan":
                    try:
                        self.current_colspan = int(v)
                    except (ValueError, TypeError):
                        pass
                elif k == "rowspan":
                    try:
                        self.current_rowspan = int(v)
                    except (ValueError, TypeError):
                        pass

    def handle_endtag(self, tag):
        if tag == "tr":
            self.rows.append(self.current_row)
        elif tag in ("td", "th"):
            self.in_cell = False
            self.current_row.append(
                {
                    "colspan": self.current_colspan,
                    "rowspan": self.current_rowspan,
                    "content": "".join(self.cell_content).strip(),
                }
            )
            self.cell_content = []

    def handle_data(self, data):
        if self.in_cell:
            self.cell_content.append(data)


def validate_table_structure(html: str) -> list[str]:
    """
    Analyzes the HTML table structure for consistency.
    Checks if all rows have the same number of columns, accounting for spans.
    """
    parser = TableGridParser()
    try:
        parser.feed(html)
    except Exception:
        return ["HTML could not be parsed."]

    if not parser.rows:
        return ["Table is empty or could not be parsed."]

    occupied = set()
    row_widths = []

    for r_idx, row in enumerate(parser.rows):
        current_col = 0
        for cell in row:
            while (r_idx, current_col) in occupied:
                current_col += 1

            c_span = cell["colspan"]
            r_span = cell["rowspan"]

            for r in range(r_span):
                for c in range(c_span):
                    occupied.add((r_idx + r, current_col + c))

            current_col += c_span
        row_widths.append(current_col)

    if not row_widths:
        return []

    width_counts = Counter(row_widths)
    if not width_counts:
        return []
    most_common_width = width_counts.most_common(1)[0][0]

    issues = []
    for r_idx, width in enumerate(row_widths):
        if width != most_common_width:
            issues.append(
                f"Row {r_idx + 1} has an effective width of {width} columns, verify in the table image how it should be."
            )

    return issues


class TableErrorAnalyzer:
    """
    Analyzes HTML tables for structural, content, and semantic errors
    based on heuristics and optional OCR data.
    """

    def __init__(self, html: str, image: Image.Image | None = None):
        self.html = html
        self.image = image
        self.issues: list[str] = []
        self.ocr_text: str | None = None

    def check_structure(self) -> None:
        """Heuristic 1: Check if all rows have the same number of columns."""
        structure_issues = validate_table_structure(self.html)
        if structure_issues:
            self.issues.append("Structure Mismatch:")
            self.issues.extend([f"- {i}" for i in structure_issues])

    def check_ocr_alignment(self) -> None:
        """Heuristic 2: Check if text aligns with OCR extraction."""
        if not self.image:
            return

        if not HAS_OCR:
            print(f"{inspect.currentframe().f_code.co_name}: OCR check disabled.", flush=True)
            # self.issues.append("Note: OCR check skipped (libraries not found).")
            return

        try:
            ocr_text = pytesseract.image_to_string(self.image)
            self.ocr_text = ocr_text

            html_clean = re.sub(r"<[^>]+>", " ", self.html)
            html_tokens = set(w.lower() for w in html_clean.split() if len(w) > 3)
            ocr_tokens = set(w.lower() for w in ocr_text.split() if len(w) > 3)

            missing_in_ocr = html_tokens - ocr_tokens

            if len(missing_in_ocr) > 0:
                sample = list(missing_in_ocr)[:5]
                self.issues.append(
                    f"Content Warning: The following words appear in HTML but were not found in OCR (possible hallucination): {', '.join(sample)}"
                )

            missing_in_html = ocr_tokens - html_tokens
            if len(missing_in_html) > 0:
                sample = list(missing_in_html)[:5]
                self.issues.append(
                    f"Content Warning: The following words appear in the image but not in the HTML (possible missing data): {', '.join(sample)}"
                )

        except Exception as e:
            print(
                f"{inspect.currentframe().f_code.co_name}: OCR check failed for table correction: {e}",
                flush=True,
            )

    def check_semantics(self) -> None:
        """Heuristic 3: Check column semantic consistency."""
        headers = re.findall(r"<th[^>]*>(.*?)</th>", self.html, re.IGNORECASE | re.DOTALL)
        rows = re.findall(r"<tr[^>]*>(.*?)</tr>", self.html, re.IGNORECASE | re.DOTALL)

        if not headers or not rows:
            return

        for col_idx, header in enumerate(headers):
            header_text = re.sub(r"<[^>]+>", "", header).strip().lower()

            if "date" in header_text or "year" in header_text:
                mismatches = 0
                checked = 0
                for row in rows[1:]:  # Skip header row if inside rows list
                    cells = re.findall(r"<td[^>]*>(.*?)</td>", row, re.IGNORECASE | re.DOTALL)
                    if col_idx < len(cells):
                        cell_text = re.sub(r"<[^>]+>", "", cells[col_idx]).strip()
                        if not cell_text:
                            continue

                        if not re.search(r"\d", cell_text):
                            mismatches += 1
                        checked += 1

                if checked > 0 and (mismatches / checked) > 0.5:
                    self.issues.append(
                        f"Semantic Warning: Column '{header_text}' seems to contain non-date/numeric data."
                    )

    def check_separators(self) -> None:
        """Heuristic 4: Check for potential missed separators in OCR content."""
        html_clean = re.sub(r"<[^>]+>", " ", self.html)

        if "!" in html_clean:
            separator_count = html_clean.count("!")

            if separator_count > 2:
                self.issues.append(
                    "Separator Warning: The character '!' appears in the image more than by chance, which indicates that this is the column separator. "
                    "! is effectively a cell separator such as |, which means that the table is a markdown that uses the ! as column separator as | does in markdown tables, beware of keeping the same empty columns as they appear in the image. There are more than one empty consecutive cells, you cannot miss them. Footer rows might not be precisely aligned, pay extra attention and give priority to the header, if present, and the body of the table. ! should not appear in the final HTML. Use the image as reference to find the cell separator.\n"
                )

    def run_analysis(self) -> str:
        self.check_structure()
        self.check_ocr_alignment()
        self.check_semantics()
        self.check_separators()

        if not self.issues:
            return ""
        return "\n".join(self.issues)


def _analyze_table_heuristic(html: str, image: Image.Image) -> str:
    print(
        "run_table_correction_process: Running table correction heuristic analysis...", flush=True
    )
    analyzer = TableErrorAnalyzer(html, image)
    return analyzer.run_analysis()


async def run_table_correction_process(
    html_content: str,
    image: Image.Image,
    executor: Any | None = None,
) -> tuple[dict[str, Any], int, int]:
    """Main entry point for table correction."""

    _, user_msg = get_table_correction_prompt_messages()

    loop = asyncio.get_running_loop()
    report = await loop.run_in_executor(executor, _analyze_table_heuristic, html_content, image)

    if not report:
        return {}, 0, 0

    print(
        f"{inspect.currentframe().f_code.co_name}: Table correction analysis Report:\n{report}",
        flush=True,
    )

    print(f"{inspect.currentframe().f_code.co_name}: Correcting table...", flush=True)

    prompt = user_msg.format(html_input=html_content, error_report=report)

    json_schema = json.dumps(
        {
            "type": "object",
            "properties": {
                "corrected_html": {"type": "string", "contentMediaType": "text/html"},
                "explanation": {"type": "string"},
            },
            "required": ["corrected_html", "explanation"],
        }
    )

    try:
        response_text, input_tokens, output_tokens = await run_clients(
            user_prompt=prompt,
            images=[image],
            models=[_make_gemini_call],
            job_type="structured_extraction",
            json_schema=json_schema,
        )

        # Clean up response
        cleaned_text = response_text.strip()
        if cleaned_text.startswith("```json"):
            cleaned_text = cleaned_text[7:]
        if cleaned_text.startswith("```"):
            cleaned_text = cleaned_text[3:]
        if cleaned_text.endswith("```"):
            cleaned_text = cleaned_text[:-3]
        cleaned_text = cleaned_text.strip()

        response = json.loads(cleaned_text)
    except Exception as e:
        raise RequestException(f"run_table_correction_process: API call failed: {e}") from e

    return response, input_tokens, output_tokens
