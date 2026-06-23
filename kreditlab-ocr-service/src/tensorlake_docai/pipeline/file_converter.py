# SPDX-License-Identifier: Apache-2.0
from pathlib import Path
import os
import subprocess
import tempfile
from typing import List, Optional
from tensorlake.applications import application, function, Retries
from tensorlake.applications import RequestError as RequestException
from tensorlake_docai.vlm.cloud import VLMExtractionTask
from tensorlake_docai.pipeline.output_formatter import format_final_output
from tensorlake_docai.extraction.structured_extraction_functions import StructuredExtraction
from tensorlake_docai.extraction.form_filling import FormFilling
from tensorlake_docai.vlm.workflow_images import file_convertion_image
from tensorlake_docai.pipeline.api import ParseRequest, Usage, QuotaResourceType
from tensorlake_docai.ocr import resolve_ocr_backend
from tensorlake_docai.pipeline.routing import (
    FILE_TYPE_MAPPING,
    download_file,
    file_convertor_should_go_to_output_formatter,
    file_convertor_should_go_to_vlm_extraction,
    file_convertor_should_go_to_structured_extraction,
)
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import (
    DocumentLayout,
    PageLayout,
    PageLayoutElement,
    PageFragmentType,
)

SECRETS = [
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_REGION",
    # "AWS_SESSION_TOKEN", # only for local test
]

# File type constants
EXCEL_EXTENSIONS = ["xlsx", "xls", "xlsm"]
EXCEL_MIME_TYPES = [
    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",  # .xlsx
    "application/vnd.ms-excel",  # .xls
    "application/vnd.ms-excel.sheet.macroenabled.12",  # .xlsm
]

# Supported file extensions that don't need conversion
SUPPORTED_EXTENSIONS = {
    "pdf",
    "jpg",
    "jpeg",
    "png",
    "txt",
    "html",
    "csv",
    "xlsx",
    "xls",
    "xlsm",
    "xml",
    "tif",
    "tiff",
    "md",
    "docx",
    "doc",
    "p7m",
}

# Text file MIME types
TEXT_MIME_TYPES = {
    "text/plain",
    "text/html",
    "text/csv",
    "application/xml",
    "text/xml",
    "text/markdown",
}

# Word document constants
DOCX_EXTENSIONS = ["docx"]
DOC_EXTENSIONS = ["doc"]
DOCX_MIME_TYPES = ["application/vnd.openxmlformats-officedocument.wordprocessingml.document"]
DOC_MIME_TYPES = ["application/msword"]

# P7M (PKCS#7) signed/encrypted document constants
P7M_EXTENSIONS = ["p7m"]
P7M_MIME_TYPES = [
    "application/pkcs7-mime",
    "application/x-pkcs7-mime",
    "application/pkcs7-signature",
]


def detect_mime_type_from_content(
    file_bytes: bytes, file_name: str = None, url_content_type: str = None
) -> str:
    """Detect MIME type from file content using magic library."""
    try:
        import magic

        # Use magic library for comprehensive MIME type detection
        detected_mime = magic.from_buffer(file_bytes, mime=True)

        file_extension = FILE_TYPE_MAPPING.get(detected_mime, None)

        # If magic didn't work, fallback to filename guessing
        if file_extension is None and file_name:
            import mimetypes

            detected_mime, _ = mimetypes.guess_type(file_name)
            file_extension = FILE_TYPE_MAPPING.get(detected_mime, None)

        if file_extension is None and url_content_type:
            detected_mime = url_content_type

        if detected_mime is None:
            raise RequestException(
                message="unable to detect a supported file type. "
                "Please provide a file name or content type while uploading the file. "
                "If it was a pre-signed url, it didn't either have a Content-Type. "
                "We tried detecting the file type using python-magic as well."
                "Docs for file uploads: https://docs.tensorlake.ai/api-reference/v2/files/upload"
                "Please email support@tensorlake.ai if you need help."
            )

        return detected_mime
    except Exception as e:
        import traceback

        print(traceback.format_exc())
        print(f"DEBUG: MIME detection failed: {e}")
        raise RequestException(
            message="Unable to detect file type. Please upload a valid PDF, image, text, or Excel file. Error: "
            + str(e)
        )


def is_excel_file(mime_type: str) -> bool:
    """Check if a file is an Excel file based on name and MIME type."""
    return mime_type in EXCEL_MIME_TYPES


def is_docx_file(mime_type: str) -> bool:
    """Check if a file is a DOCX file based on name and MIME type."""
    return mime_type in DOCX_MIME_TYPES


def is_doc_file(mime_type: str) -> bool:
    """Check if a file is a DOC file based on name and MIME type."""
    return mime_type in DOC_MIME_TYPES


def is_p7m_file(mime_type: str, file_name: str = "") -> bool:
    """Check if a file is a P7M (PKCS#7) file based on MIME type and filename."""
    return mime_type in P7M_MIME_TYPES or (file_name and file_name.lower().endswith(".p7m"))


def needs_conversion(file_extension: str) -> bool:
    """Check if a file extension needs to be converted to PDF."""
    return file_extension not in SUPPORTED_EXTENSIONS


def is_text_file(mime_type: str) -> bool:
    """Check if a file is a text file based on MIME type."""
    return mime_type.startswith("text/") or mime_type in TEXT_MIME_TYPES


def get_excel_engine(file_name: str) -> str:
    """Get the appropriate pandas engine for reading Excel files."""
    if file_name.lower().endswith(".xls"):
        return "xlrd"
    else:  # .xlsx, .xlsm
        return "openpyxl"


def count_excel_sheets(file_bytes: bytes, file_name: str) -> int:
    """Count the number of sheets in an Excel file."""
    try:
        import pandas as pd
        import tempfile
        from pathlib import Path

        # Create a temporary file to write the Excel bytes
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_file.flush()
            temp_path = Path(temp_file.name)

        # Get appropriate engine and count sheets
        engine = get_excel_engine(file_name)
        sheets = pd.read_excel(temp_path, sheet_name=None, engine=engine)
        sheet_count = len(sheets)

        # Clean up the temporary file
        temp_path.unlink()

        return sheet_count
    except Exception as e:
        print(f"DEBUG: Failed to count Excel sheets: {e}")
        raise RequestException(
            message="Unable to process Excel document. Please ensure it is a valid Excel file and not password-protected. Error: "
            + str(e)
        )


def process_excel_from_bytes(file_bytes: bytes, file_name: str) -> list[PageLayout]:
    """Process Excel file from bytes by creating a temporary file and processing sheets."""
    try:
        import tempfile
        from pathlib import Path

        # Create a temporary file to write the Excel bytes
        with tempfile.NamedTemporaryFile(suffix=".xlsx", delete=False) as temp_file:
            temp_file.write(file_bytes)
            temp_file.flush()
            temp_path = Path(temp_file.name)

        # Process the Excel file
        pages = process_excel_sheets(temp_path)

        # Clean up the temporary file
        temp_path.unlink()

        return pages
    except Exception as e:
        print(f"DEBUG: Failed to process Excel from bytes: {e}")
        raise RequestException(
            message="Unable to process Excel document. Please ensure it is a valid Excel file and not password-protected. Error: "
            + str(e)
        )


def count_docx_pages(file_bytes: bytes) -> int:
    """Count the number of pages in a DOCX file."""
    try:
        import zipfile
        import xml.etree.ElementTree as ET
        import io

        with zipfile.ZipFile(io.BytesIO(file_bytes)) as zf:
            if "docProps/app.xml" in zf.namelist():
                app_xml = zf.read("docProps/app.xml")
                root = ET.fromstring(app_xml)

                # Extended properties namespace
                ns = "http://schemas.openxmlformats.org/officeDocument/2006/extended-properties"
                # ElementTree find with namespace
                pages = root.find(f"{{{ns}}}Pages")

                if pages is not None and pages.text:
                    return int(pages.text)

                # Fallback search
                for child in root:
                    if child.tag.endswith("Pages") and child.text:
                        return int(child.text)

    except Exception as e:
        print(f"DEBUG: Failed to count DOCX pages: {e}")

    return 1


def count_document_pages(file_bytes: bytes, mime_type: str, file_name: str = "") -> int:
    """
    Count the number of pages in a document.
    Supports PDF, Excel, and various image formats.
    """
    print(f"DEBUG: count_document_pages called with mime_type: {mime_type}, file_name: {file_name}")

    # Prefer TIFF detection by filename or mime type
    if (file_name and file_name.lower().endswith(("tif", "tiff"))) or mime_type in (
        "image/tiff",
        "image/tif",
    ):
        try:
            from PIL import Image
            import io

            img = Image.open(io.BytesIO(file_bytes))
            page_count = getattr(img, "n_frames", 1)
            try:
                img.close()
            except Exception:
                pass
            return int(page_count)
        except Exception as e:
            print(f"DEBUG: Failed to count TIFF pages via PIL: {e}")
            raise RequestException(
                message="Unable to open image document. Please ensure the file is a valid TIFF or use PDF. Error: "
                + str(e)
            )
    if mime_type == "application/pdf":
        try:
            import io
            from pypdf import PdfReader

            return len(PdfReader(io.BytesIO(file_bytes)).pages)

        except Exception as e:
            print(f"DEBUG: Failed to read PDF for page counting: {e}")
            raise RequestException(
                message="Unable to open the PDF document. Please ensure the file is a valid, non-corrupted PDF. Error: "
                + str(e)
            )
    elif mime_type.startswith("image/"):
        # For image files, treat as single page
        print("DEBUG: Treating as image file")
        return 1
    elif mime_type.startswith("text/") or mime_type in ["application/xml", "text/xml"]:
        # For text files, treat as single page
        print("DEBUG: Treating as text file")
        return 1
    else:
        # Check if this is an Excel file
        if is_excel_file(mime_type):
            # For Excel files, count the number of sheets
            print("DEBUG: Treating as Excel file")
            return count_excel_sheets(file_bytes, file_name)
        elif is_docx_file(mime_type):
            print("DEBUG: Treating as DOCX file")
            return count_docx_pages(file_bytes)
        else:
            # For other file types, try PyMuPDF
            print("DEBUG: Trying PyMuPDF for unknown file type")
            try:
                import fitz  # PyMuPDF

                doc = fitz.open(stream=file_bytes)
                page_count = doc.page_count
                doc.close()
                print(f"DEBUG: PyMuPDF returned {page_count} pages")
                return page_count
            except Exception:
                # If PyMuPDF can't handle it, assume it's a single page
                print("DEBUG: PyMuPDF failed, defaulting to 1 page")
                return 1


def validate_quota(request: ParseRequest, total_document_pages: int) -> None:
    """
    Validate that the request doesn't exceed any quota limits.
    Raises an exception if any quota is exceeded.

    Note: A remaining_quota value of -1 indicates unlimited quota (no validation).
    """
    if request.org_quota is None:
        # No quota to validate
        return

    # Determine the number of pages to be processed
    if request.pages_to_parse is not None:
        pages_to_process = len(request.pages_to_parse)
    else:
        pages_to_process = total_document_pages

    # Check each quota
    for quota in request.org_quota.quotas:
        # Skip validation if quota is unlimited (-1)
        if quota.remaining_quota == -1:
            print(
                f"Skipping quota validation for {quota.resource_type.value} because it is unlimited"
            )
            continue

        if quota.resource_type == QuotaResourceType.PAGES_PARSED:
            if pages_to_process > quota.remaining_quota:
                raise RequestException(
                    message=f"Quota exceeded: Requested {pages_to_process} pages but only "
                    f"{quota.remaining_quota} pages remaining for {quota.resource_type.value}"
                )

        elif quota.resource_type == QuotaResourceType.SIGNATURE_DETECTION:
            if request.detect_signature and pages_to_process > quota.remaining_quota:
                raise RequestException(
                    message=f"Quota exceeded: Signature detection requested for {pages_to_process} pages "
                    f"but only {quota.remaining_quota} pages remaining for {quota.resource_type.value}"
                )


def convert_to_pdf(file_path: Path) -> Path:
    temp_dir = Path(tempfile.gettempdir())

    cmd = [
        "soffice",
        "--headless",
        "--convert-to",
        "pdf",
        "--outdir",
        str(temp_dir),
        str(file_path),
    ]

    try:
        res = subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        if res.returncode != 0:
            raise RequestException(message="Unable to convert to PDF file")

        pdf_path = file_path.with_suffix(".pdf")

        print(f"Creating new file with name {pdf_path.name}")

        if not pdf_path.exists():
            raise RequestException(message="Unable to convert to PDF file")

        return pdf_path

    except FileNotFoundError as e:
        raise RequestException(
            message="Document processing failed during ingestion. Unable to convert to PDF file from the source file. Please upload a PDF directly or try again later. Error: "
            + str(e)
        )
    except subprocess.CalledProcessError as e:
        raise RequestException(
            message="Document processing failed during ingestion. Unable to convert to PDF file from the source file. Please ensure the document is supported and not password-protected. Error: "
            + str(e)
        )
    except Exception as e:
        print(f"DEBUG: Unexpected error during PDF conversion: {e}")
        raise RequestException(
            message="Document processing failed during ingestion. Unable to convert to PDF file from the source file. Please try again or contact Tensorlake support with the trace ID of the job. Error: "
            + str(e)
        )


def extract_p7m_content(file_bytes: bytes) -> bytes:
    """
    Extract the signed/encrypted content from a P7M (PKCS#7) file.
    Returns the inner document bytes.
    """
    import subprocess

    input_path = None
    output_path = None

    try:
        # Create temp files
        with tempfile.NamedTemporaryFile(suffix=".p7m", delete=False) as temp_input:
            temp_input.write(file_bytes)
            temp_input.flush()
            input_path = Path(temp_input.name)

        with tempfile.NamedTemporaryFile(suffix=".extracted", delete=False) as temp_output:
            output_path = Path(temp_output.name)

        # Try multiple approaches to extract P7M content
        commands_to_try = [
            # CMS (most modern, handles most formats)
            [
                "openssl",
                "cms",
                "-verify",
                "-noverify",
                "-in",
                str(input_path),
                "-inform",
                "DER",
                "-out",
                str(output_path),
            ],
            [
                "openssl",
                "cms",
                "-verify",
                "-noverify",
                "-in",
                str(input_path),
                "-inform",
                "PEM",
                "-out",
                str(output_path),
            ],
            # SMIME (for email-style signed messages)
            [
                "openssl",
                "smime",
                "-verify",
                "-noverify",
                "-in",
                str(input_path),
                "-inform",
                "DER",
                "-out",
                str(output_path),
            ],
            [
                "openssl",
                "smime",
                "-verify",
                "-noverify",
                "-in",
                str(input_path),
                "-inform",
                "PEM",
                "-out",
                str(output_path),
            ],
            # PKCS7 print (just extract the data)
            [
                "openssl",
                "pkcs7",
                "-print_certs",
                "-in",
                str(input_path),
                "-inform",
                "DER",
                "-out",
                str(output_path),
            ],
        ]

        result = None
        for cmd in commands_to_try:
            result = subprocess.run(cmd, capture_output=True, timeout=10)
            if result.returncode == 0:
                extracted_bytes = output_path.read_bytes()
                if len(extracted_bytes) > 0:
                    print(
                        f"Successfully extracted {len(extracted_bytes)} bytes from P7M file using: {' '.join(cmd[:3])}"
                    )
                    return extracted_bytes

        # If all attempts failed
        error_msg = result.stderr.decode("utf-8", errors="ignore") if result else "Unknown error"
        raise RequestException(
            message=f"Unable to extract content from P7M file. The file may be encrypted or corrupted. Error: {error_msg}"
        )

    except subprocess.TimeoutExpired:
        raise RequestException(
            message="P7M file extraction timed out. The file may be too large or corrupted."
        )
    except FileNotFoundError:
        raise RequestException(
            message="OpenSSL not found. Unable to process P7M files. Please contact Tensorlake support."
        )
    except Exception as e:
        print(f"DEBUG: Failed to extract P7M content: {e}")
        import traceback

        print(traceback.format_exc())
        raise RequestException(
            message=f"Unable to process P7M file. The file may be encrypted, password-protected, or corrupted. Error: {str(e)}"
        )
    finally:
        # Clean up temp files
        if input_path and input_path.exists():
            input_path.unlink()
        if output_path and output_path.exists():
            output_path.unlink()


def convert_doc_to_docx(file_path: Path) -> Path:
    """Convert DOC file to DOCX using LibreOffice."""
    temp_dir = Path(tempfile.gettempdir())

    cmd = [
        "soffice",
        "--headless",
        "--convert-to",
        "docx",
        "--outdir",
        str(temp_dir),
        str(file_path),
    ]

    try:
        res = subprocess.run(cmd, check=True, capture_output=True, timeout=60)
        if res.returncode != 0:
            raise RequestException(message="Unable to convert DOC to DOCX file")

        docx_path = file_path.with_suffix(".docx")

        print(f"Creating new DOCX file with name {docx_path.name}")

        if not docx_path.exists():
            raise RequestException(message="Unable to convert DOC to DOCX file")

        return docx_path

    except FileNotFoundError:
        raise RequestException(
            message="Document processing failed during conversion. LibreOffice not found. Please ensure LibreOffice is installed. Error: LibreOffice not found"
        )
    except subprocess.CalledProcessError as e:
        raise RequestException(
            message="Document processing failed during DOC to DOCX conversion. Please ensure the document is supported and not password-protected. Error: "
            + str(e)
        )
    except Exception as e:
        print(f"DEBUG: Unexpected error during DOC to DOCX conversion: {e}")
        raise RequestException(
            message="Document processing failed during DOC to DOCX conversion. Please try again or contact Tensorlake support with the trace ID of the job. Error: "
            + str(e)
        )


def process_excel_sheets(file_path: Path) -> list[PageLayout]:
    """Process Excel file by converting all sheets to HTML and Markdown."""
    import pandas as pd
    from markdownify import markdownify

    # Use appropriate engine based on file extension
    engine = get_excel_engine(file_path.name)

    sheets = pd.read_excel(file_path, sheet_name=None, engine=engine)
    pages = []

    for page_num, (sheet_name, df) in enumerate(sheets.items(), 1):
        # Convert to HTML first
        html_content = df.fillna("").to_html(
            index=False, classes="excel-table", table_id=f"sheet-{page_num}"
        )

        # Clean up the HTML
        cleaned_tables = clean_and_split_html_tables(html_content)

        # Create elements for this page
        elements = []
        reading_order = 1

        for table_idx, html_table in enumerate(cleaned_tables):
            # Convert HTML table to markdown
            markdown_table = markdownify(html_table)

            # Create table element with both HTML and Markdown representations
            table_element = PageLayoutElement(
                bbox=[0, 0, 0, 0],
                fragment_type=PageFragmentType.TABLE,
                ocr_text=markdown_table,
                score=1.0,
                reading_order=reading_order,
            )
            # Store both representations
            table_element.markdown = markdown_table
            table_element.html = html_table

            elements.append(table_element)
            reading_order += 1

        # Create page layout with sheet name as page_class
        page_layout = PageLayout(
            page_number=page_num,
            elements=elements,
            shape=(1000, 1000),
            page_class=sheet_name,
            page_dimensions={"width": 1000, "height": 1000},
        )
        pages.append(page_layout)

    return pages


def clean_and_split_html_tables(html_content: str) -> List[str]:
    """Simple splitting of HTML table at empty rows."""
    from bs4 import BeautifulSoup, Tag
    import copy

    def is_empty_row(row: Tag) -> bool:
        """Check if row is completely empty (no meaningful text content)."""
        cells = row.find_all(["td", "th"])
        if not cells:
            return True
        # Check if all cells are empty or contain only whitespace/HTML tags
        for cell in cells:
            text = cell.get_text(strip=True)
            if text and text not in ["", "&nbsp;", "\xa0"]:  # Include common empty placeholders
                return False
        return True

    def clean_unnamed_cells(row: Tag) -> Tag:
        """Clean up 'Unnamed: X' cells in a row."""
        cleaned_row = copy.deepcopy(row)
        cells = cleaned_row.find_all(["td", "th"])
        for cell in cells:
            text = cell.get_text(strip=True)
            if text.startswith("Unnamed:"):
                cell.clear()
                cell.string = ""  # Replace with empty string
        return cleaned_row

    soup = BeautifulSoup(html_content, "html.parser")
    original_table = soup.find("table")
    if not original_table:
        return [html_content]

    table_class = original_table.get("class", [])
    all_rows = original_table.find_all("tr")

    print(f"Processing {len(all_rows)} total rows")

    # Split at empty rows
    all_chunks = []
    current_chunk = []

    for i, row in enumerate(all_rows):
        if is_empty_row(row):
            print(f"Found empty row at index {i}")
            if current_chunk:
                all_chunks.append(current_chunk)
                current_chunk = []
        else:
            current_chunk.append(row)

    # Add the last chunk if it exists
    if current_chunk:
        all_chunks.append(current_chunk)

    print(f"Split into {len(all_chunks)} chunks")

    # Create separate tables for each chunk
    final_tables_html = []
    for chunk_idx, chunk in enumerate(all_chunks):
        if chunk:  # Only process non-empty chunks
            print(f"Creating table for chunk {chunk_idx} with {len(chunk)} rows")

            # Create new table
            table = BeautifulSoup("<table></table>", "html.parser").find("table")

            if table_class:
                table["class"] = table_class

            # Add all rows from this chunk, cleaning unnamed cells
            for row in chunk:
                cleaned_row = clean_unnamed_cells(row)
                table.append(cleaned_row)

            final_tables_html.append(str(table))

    print(f"Created {len(final_tables_html)} final tables")
    return final_tables_html if final_tables_html else [html_content]


def process_file_from_s3_or_url(request: ParseRequest) -> Optional[bytes]:
    """Process file from S3 or URL (when file_bytes is None).
    Returns PDF bytes if file was converted to PDF, None otherwise."""
    print("DEBUG: Processing file from S3/URL")
    file_data = download_file(request)
    detected_mime = detect_mime_type_from_content(
        file_data.file_bytes, request.file_name, file_data.content_type
    )
    request.mime_type = detected_mime
    request.file_bytes = file_data.file_bytes
    converted_pdf_bytes = None

    # Handle P7M files - extract inner content first
    if is_p7m_file(detected_mime, request.file_name):
        print("DEBUG: Detected P7M file, extracting content")
        request.file_bytes = extract_p7m_content(request.file_bytes)

        # Re-detect MIME type of extracted content
        detected_mime = detect_mime_type_from_content(request.file_bytes, request.file_name, None)
        request.mime_type = detected_mime
        print(f"DEBUG: Extracted content MIME type: {detected_mime}")

        # If extracted content is a PDF, store it as converted output
        if detected_mime == "application/pdf":
            converted_pdf_bytes = request.file_bytes

    file_extension = FILE_TYPE_MAPPING.get(detected_mime, None)
    print(f"DEBUG: File extension: {file_extension}")
    if file_extension is None:
        raise RequestException(
            message=f"File extension couldn't be determined for file: {request.file_name}"
        )
    if needs_conversion(file_extension) or is_doc_file(detected_mime):
        with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=True) as temp_file:
            temp_file.write(request.file_bytes)
            temp_file.flush()
            temp_path = Path(temp_file.name)

            if is_doc_file(detected_mime):
                print("DEBUG: Converting DOC to DOCX")
                file_path = convert_doc_to_docx(temp_path)
                request.mime_type = (
                    "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
                )
            # Convert file to PDF if needed (but not DOCX files)
            elif needs_conversion(file_extension) and not is_docx_file(detected_mime):
                print(f"DEBUG: Converting {temp_path} to PDF")
                file_path = convert_to_pdf(temp_path)
                request.mime_type = "application/pdf"
                converted_pdf_bytes = file_path.read_bytes()
            else:
                print("DEBUG: File format supported, skipping conversion")

            request.file_bytes = file_path.read_bytes()
            request.file_name = file_path.name
            os.unlink(file_path)

    return converted_pdf_bytes


def create_pages_from_content(request: ParseRequest) -> tuple[list[PageLayout], Optional[str]]:
    """Create page layouts based on file content type. Returns (pages, docx_pdf_base64)."""
    import chardet

    pages = []
    docx_pdf_base64 = None

    # Handle text files
    if is_text_file(request.mime_type):
        print("DEBUG: Processing as text file")
        # Preserve text/csv for structured extraction, normalize others to text/plain
        if request.mime_type != "text/csv":
            request.mime_type = "text/plain"
        try:
            text_content = request.file_bytes.decode("utf-8-sig")  # handles UTF-8 with/without BOM
        except UnicodeDecodeError as decode_error:
            detected = chardet.detect(request.file_bytes[:10_000])
            encoding = detected.get("encoding")
            confidence = detected.get("confidence", 0.0)

            if encoding:
                encoding = encoding.lower()

            print(f"DEBUG: Detected encoding: {encoding}, confidence: {confidence}")

            _utf8_variants = {"utf-8", "utf_8", "utf-8-sig", "utf_8_sig", "ascii"}
            if encoding and encoding not in _utf8_variants and confidence > 0.5:
                try:
                    text_content = request.file_bytes.decode(encoding)
                except (UnicodeDecodeError, LookupError) as fallback_error:
                    print(f"DEBUG: Fallback text decode failed: {fallback_error}")
                    raise RequestException(
                        message=(
                            "We couldn’t read this file due to an unsupported text encoding. "
                            "Please re-save the file as UTF-8 and upload again."
                        )
                    ) from fallback_error
            else:
                raise RequestException(
                    message=(
                        "We couldn’t read this file due to an unsupported text encoding. "
                        "Please re-save the file as UTF-8 and upload again."
                    )
                ) from decode_error

        page_layout = PageLayout(
            page_number=1,
            elements=[
                PageLayoutElement(
                    bbox=[0, 0, 0, 0],
                    fragment_type=PageFragmentType.TEXT,
                    ocr_text=text_content,
                    score=1.0,
                    reading_order=1,
                )
            ],
            shape=(1, 1),
            page_dimensions={"width": 1, "height": 1},
        )
        pages.append(page_layout)

    # Handle Excel files
    elif is_excel_file(request.mime_type):
        print("DEBUG: Processing as Excel file")
        pages = process_excel_from_bytes(request.file_bytes, request.file_name)

        request.mime_type = "text/table"  # Set mime_type to text/table since we're converting everything to text table
        print(f"Processed Excel file with {len(pages)} sheets (all converted to text)")

    # Handle DOCX files
    elif is_docx_file(request.mime_type):
        print("DEBUG: Processing as DOCX file")
        from tensorlake_docai.pipeline.docx_parsing import process_docx_to_structured_pages
        import base64

        # Process DOCX to get structured pages and PDF bytes
        page_items, pdf_bytes = process_docx_to_structured_pages(request.file_bytes)

        # Create pages with bboxes
        for page_num in sorted(page_items.keys()):
            elements = []
            for idx, item in enumerate(sorted(page_items[page_num], key=lambda x: x["bbox"][1]), 1):
                element = PageLayoutElement(
                    bbox=item["bbox"],
                    fragment_type=item["fragment_type"],  # Use determined fragment type
                    ocr_text=item["content"],  # Use cleaned content (preserves semantic tags)
                    score=1.0,
                    reading_order=idx,
                )
                # Store original HTML for reference
                element.html = item["html"]
                # For tables, store HTML in markdown field as well
                if item["fragment_type"] == PageFragmentType.TABLE:
                    element.markdown = item["html"]
                elements.append(element)

            page_layout = PageLayout(
                page_number=page_num + 1,  # 0-indexed to 1-indexed
                elements=elements,
                shape=(1, 1),
                page_dimensions={"width": 1, "height": 1},
            )
            pages.append(page_layout)

        # Store base64 PDF for output
        docx_pdf_base64 = base64.b64encode(pdf_bytes).decode("utf-8")
        base64_size_kb = len(docx_pdf_base64) / 1024
        print(
            f"Base64 encoded PDF size in response: {base64_size_kb:.2f} KB ({len(docx_pdf_base64)} chars)"
        )

        request.mime_type = "text/html"  # Set mime_type to text/html since we're returning HTML
        print(f"Processed DOCX file with {len(pages)} pages and bboxes extracted")

    return pages, docx_pdf_base64


@application()
@function(
    description="Convert documents as needed. Process DOCX files directly with tracked changes, convert DOC to DOCX, convert other formats to PDF. Upload new file to S3.",
    image=file_convertion_image,
    secrets=SECRETS,
    timeout=30 * 60,  # 30 minutes
    cpu=2,
    memory=5,
    # This function is using /tmp ephemeral disk space
    ephemeral_disk=10,
    retries=Retries(max_retries=2),
    max_containers=200,
    min_containers=int(os.getenv("TENSORLAKE_MIN_CONTAINERS", "0")),
)
# dict is ParsedDocumentRef converted to dict.
def normalize_file_type_and_upload(raw_request: dict) -> ParseResult | dict:
    print(f"DEBUG: raw_request keys: {list(raw_request.keys())}")
    print(f"DEBUG: raw_request mime_type: {repr(raw_request.get('mime_type'))}")
    print(f"DEBUG: raw_request file_name: {repr(raw_request.get('file_name'))}")
    print(f"DEBUG: raw_request file_url: {repr(raw_request.get('file_url'))}")
    print(
        f"DEBUG: raw_request file_bytes: {repr(raw_request.get('file_bytes', 'NOT_PROVIDED')[:50] if raw_request.get('file_bytes') else None)}"
    )
    print(f"DEBUG: raw_request form_filling: {repr(raw_request.get('form_filling'))}")

    request = ParseRequest.model_validate(raw_request)
    print(f"DEBUG: request.form_filling: {repr(request.form_filling)}")
    print(f"DEBUG: request keys: {list(request.model_dump().keys())}")
    converted_pdf_bytes = process_file_from_s3_or_url(request)
    print(f"DEBUG: request file_bytes: {len(request.file_bytes)}")
    try:
        total_document_pages = count_document_pages(
            request.file_bytes, request.mime_type, request.file_name
        )
        print(f"Document has {total_document_pages} total pages")
    except RequestException:
        # Re-raise RequestException to stop processing invalid files
        raise
    except Exception as e:
        print(f"Warning: Could not count document pages: {e}")
        total_document_pages = 1  # Default to 1 page if counting fails

    # Validate quotas before processing
    validate_quota(request, total_document_pages)

    # Validate that page_classes isn't used without page classification being enabled
    se_requests = request.structured_extraction_requests or []
    for se_req in se_requests:
        if se_req.page_classes and not request.page_classification_request:
            raise RequestException(
                message=(
                    f"structured_extraction_request specifies page_classes "
                    f"{se_req.page_classes!r} but page_classification_request is not set. "
                    f"Enable page classification in your request before filtering by page_classes."
                )
            )

    # Validate requested page range (if provided) against total pages detected
    if request.pages_to_parse:
        valid_pages = [p for p in request.pages_to_parse if 1 <= p <= total_document_pages]
        invalid_pages = [p for p in request.pages_to_parse if p < 1 or p > total_document_pages]

        if invalid_pages:
            print(
                f"Warning: Ignoring invalid pages {invalid_pages} (document has {total_document_pages} pages)"
            )

        if not valid_pages:
            first_invalid = sorted(invalid_pages)[0]
            raise RequestException(
                message=(
                    f"Invalid page range specified. Document has {total_document_pages} pages, "
                    f"but page {first_invalid} was requested. Please specify pages 1-{total_document_pages}."
                )
            )

        request.pages_to_parse = valid_pages
        print(f"Processing valid pages: {valid_pages}")

    # Create page layouts based on content type
    pages, docx_pdf_base64 = create_pages_from_content(request)
    print(f"DEBUG: Created {len(pages)} pages from content")

    # Use converted PDF bytes if available (from file conversion), otherwise use DOCX PDF
    import base64

    final_pdf_base64 = None
    if converted_pdf_bytes:
        final_pdf_base64 = base64.b64encode(converted_pdf_bytes).decode("utf-8")
        print(f"Using converted PDF base64 ({len(final_pdf_base64)} chars)")
    elif docx_pdf_base64:
        final_pdf_base64 = docx_pdf_base64

    usage = Usage(
        pages_parsed=len(pages),
        extraction_input_tokens_used=0,
        extraction_output_tokens_used=0,
        summarization_input_tokens_used=0,
        summarization_output_tokens_used=0,
        header_correction_input_tokens_used=0,
        header_correction_output_tokens_used=0,
    )
    print(f"Parsed {usage.pages_parsed} pages")
    parse_result = ParseResult(
        document_layout=DocumentLayout(
            pages=pages, scale_factor=1, total_pages=total_document_pages
        ),
        request=request,
        usage=usage,
        docx_converted_pdf_base64=final_pdf_base64,
    )

    # Node-by-node routing decisions
    if request.form_filling:
        print("🔀 FILE_CONVERTOR → FormFilling")
        return FormFilling().run.future(parse_result)

    if file_convertor_should_go_to_output_formatter(request):
        print("🔀 FILE_CONVERTOR → OutputFormatter (text file, no processing needed)")
        return format_final_output(parse_result)

    elif file_convertor_should_go_to_vlm_extraction(request):
        print("🔀 FILE_CONVERTOR → VLMExtractionTask")
        return VLMExtractionTask().run.future(parse_result)

    elif file_convertor_should_go_to_structured_extraction(request):
        print("🔀 FILE_CONVERTOR → StructuredExtraction (text file with SE)")
        return StructuredExtraction().run.future(parse_result)

    else:
        # Default: dispatch to the OCR backend selected by ``request.ocr_model``
        # (or the registry default when the field is unset). Backends are
        # responsible for short-circuiting if the file is text-only.
        backend_cls = resolve_ocr_backend(request.ocr_model)
        print(f"🔀 FILE_CONVERTOR → {backend_cls.__name__} (ocr_model={request.ocr_model!r})")
        return backend_cls().run.future(parse_result)
