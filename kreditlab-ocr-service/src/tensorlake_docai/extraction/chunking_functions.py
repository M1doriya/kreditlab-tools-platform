# SPDX-License-Identifier: Apache-2.0
from typing import List
from enum import Enum
from tensorlake_docai.pipeline.api import (
    PageFragmentType,
    ParsedDocument,
    Chunk,
    ParseRequest,
    Page,
    StructuredExtractionRequest,
    PageClass,
)
from tensorlake_docai.postprocess.formatter import (
    page_fragment_to_markdown,
    document_to_markdown,
    page_to_markdown,
    document_layout_to_document,
)
from tensorlake.applications import (
    RequestError as RequestException,
)
from tensorlake_docai.models.intermediate_objects import ParseResult
from tensorlake_docai.models.layout_objects import PageLayout


class ChunkingStrategy(Enum):
    VARIABLE = "variable"
    SECTION = "section"
    PAGE = "page"
    FRAGMENT = "fragment"
    NONE = "none"
    PATTERNS = "patterns"


def chunk_document(result: ParseResult) -> ParsedDocument:
    print(f"chunk strategy requested: {result.request.chunk_strategy}")
    chunks = []

    chunking_strategy = result.request.chunk_strategy or "none"

    # Create pages without merged tables for chunking
    pages = document_layout_to_document(
        result.document_layout.pages,
        result.document_layout.scale_factor,
        result.request.ignore_sections,
        merged_tables=None,
        chunking_strategy=chunking_strategy,
    )
    if result.request.chunk_strategy == ChunkingStrategy.FRAGMENT.value:
        chunks = fragment_chunking(pages, result.request)
    elif result.request.chunk_strategy == ChunkingStrategy.SECTION.value:
        chunks = section_chunking(pages, result.request)
    elif result.request.chunk_strategy == ChunkingStrategy.PAGE.value:
        chunks = page_chunking(pages, result.request)
    elif result.request.chunk_strategy == ChunkingStrategy.PATTERNS.value:
        # Patterns chunking in chunk_document is not supported since patterns are not available
        # This should only be used in structured extraction flow via chunk_pages
        raise RequestException("Patterns chunking is only supported in structured extraction flow")
    else:
        chunk = document_to_markdown(pages, result.request)
        chunks = [Chunk(content=chunk, page_number=0)]

    # Create pages with merged tables for document markdown
    merged_tables = result.document_layout.merged_tables if result.document_layout else None
    pages_for_markdown = document_layout_to_document(
        result.document_layout.pages,
        result.document_layout.scale_factor,
        result.request.ignore_sections,
        merged_tables=merged_tables,
        chunking_strategy="none",
    )
    document_markdown = document_to_markdown(pages_for_markdown, result.request)

    parsed_document = ParsedDocument(
        parsed_pages_count=len(pages),
        chunks=chunks,
        pages=pages,
        merged_tables=merged_tables,
        document_markdown=document_markdown,
    )
    page_classes = {}
    for page_layout in result.document_layout.pages:
        if page_layout.page_class:
            classes = page_layout.page_class
            # convert to list if it's a string
            if isinstance(classes, str):
                classes = [classes]
            for class_name in classes:
                if class_name not in page_classes:
                    page_classes[class_name] = PageClass(
                        page_numbers=[],
                        page_class=class_name,
                        classification_reasons={},
                        classification_confidences={},
                    )
                page_classes[class_name].page_numbers.append(page_layout.page_number)
                if page_layout.classification_reason:
                    page_classes[class_name].classification_reasons[
                        page_layout.page_number
                    ] = page_layout.classification_reason
                if page_layout.classification_confidence is not None:
                    page_classes[class_name].classification_confidences[
                        page_layout.page_number
                    ] = page_layout.classification_confidence

    parsed_document.page_classes = list(page_classes.values())

    return parsed_document


def chunk_pages(
    page_layouts: List[PageLayout],
    chunking_strategy: str,
    result: ParseResult,
    start_patterns: List[str] = None,
    end_patterns: List[str] = None,
    merged_tables=None,
) -> List[Chunk]:
    print(f"chunk strategy requested: {chunking_strategy}")
    chunks = []

    pages = document_layout_to_document(
        page_layouts,
        result.document_layout.scale_factor,
        result.request.ignore_sections,
        merged_tables=merged_tables,
        chunking_strategy=chunking_strategy,
    )

    if chunking_strategy == ChunkingStrategy.FRAGMENT.value:
        chunks = fragment_chunking(pages, result.request)
    elif chunking_strategy == ChunkingStrategy.SECTION.value:
        chunks = section_chunking(pages, result.request)
    elif chunking_strategy == ChunkingStrategy.PAGE.value:
        chunks = page_chunking(pages, result.request)
    elif chunking_strategy == ChunkingStrategy.PATTERNS.value:
        chunks = patterns_chunking(pages, result.request, start_patterns, end_patterns)
    else:
        chunk = document_to_markdown(pages, result.request)
        chunks = [Chunk(content=chunk, page_number=0)]

    return chunks


def page_chunking(pages: List[Page], request: StructuredExtractionRequest) -> List[Chunk]:
    page_chunks = []
    for page in pages:
        # Track all element ref_ids for this page for citation filtering
        element_ids = []
        for fragment in page.page_fragments:
            if fragment.reading_order is not None:
                element_id = f"{page.page_number}.{fragment.reading_order}"
                element_ids.append(element_id)

        page_chunks.append(
            Chunk(
                content=page_to_markdown(page, request),
                page_number=page.page_number,
                element_ids=element_ids if element_ids else None,
            )
        )
    return page_chunks


def fragment_chunking(pages: List[Page], request: ParseRequest) -> List[Chunk]:
    fragment_chunks = []
    for page in pages:
        for fragment in page.page_fragments:
            # Track element ref_id for citation filtering
            element_id = (
                f"{page.page_number}.{fragment.reading_order}"
                if fragment.reading_order is not None
                else None
            )
            element_ids = [element_id] if element_id else None
            fragment_chunks.append(
                Chunk(
                    content=page_fragment_to_markdown(fragment, request),
                    page_number=page.page_number,
                    element_ids=element_ids,
                )
            )
    return fragment_chunks


def section_chunking(pages: List[Page], request: ParseRequest) -> List[Chunk]:
    section_types = [PageFragmentType.SECTION_HEADER, PageFragmentType.FORM]
    section_chunks = []

    current_section = ""
    current_section_start_page = None
    current_section_pages = set()
    current_section_element_ids = []
    last_was_section_header = False
    for page in pages:
        for fragment in page.page_fragments:
            # If we find a section header and the last fragment wasn't a section header,
            # we add the current section to the list and start a new section
            if (
                fragment.fragment_type in section_types
                and current_section
                and not last_was_section_header
            ):
                page_list = sorted(list(current_section_pages))
                section_chunks.append(
                    Chunk(
                        content=current_section,
                        page_number=current_section_start_page,
                        page_numbers=page_list,
                        element_ids=(
                            current_section_element_ids if current_section_element_ids else None
                        ),
                    )
                )
                current_section = ""
                current_section_pages = set()
                current_section_element_ids = []

            # Add the fragment to the current section
            if not current_section:
                current_section_start_page = page.page_number
            current_section_pages.add(page.page_number)
            current_section += page_fragment_to_markdown(fragment, request)

            # Track element ref_id for citation filtering
            if fragment.reading_order is not None:
                element_id = f"{page.page_number}.{fragment.reading_order}"
                current_section_element_ids.append(element_id)

            # Update the flag for the next iteration
            last_was_section_header = fragment.fragment_type in section_types

    # If we have a current section, we add it to the list
    if current_section:
        page_list = sorted(list(current_section_pages))
        section_chunks.append(
            Chunk(
                content=current_section,
                page_number=current_section_start_page,
                page_numbers=page_list,
                element_ids=current_section_element_ids if current_section_element_ids else None,
            )
        )

    return section_chunks


def patterns_chunking(
    pages: List[Page],
    request: ParseRequest,
    start_patterns: List[str] = None,
    end_patterns: List[str] = None,
) -> List[Chunk]:
    """
    Chunk document into blocks based on user-defined start/end patterns.

    Simple rules:
    - If only start_patterns: Each match starts a new chunk, includes the start line
    - If only end_patterns: Each match ends current chunk, includes the end line
    - If both: start_patterns begin chunks, end_patterns end them
    - If neither: Return whole document as one chunk
    """
    import re

    start_patterns = start_patterns or []
    end_patterns = end_patterns or []
    print("Using patterns chunking strategy")
    print(f"start_patterns: {start_patterns}")
    print(f"end_patterns: {end_patterns}")

    # If no patterns provided, return whole document as one chunk
    if not start_patterns and not end_patterns:
        raise RequestException(
            "Please provide at least one start or end pattern for anchor-based chunking"
        )

    # Compile patterns (case-insensitive by default)
    start_regexes = [re.compile(pattern, re.IGNORECASE) for pattern in start_patterns]
    end_regexes = [re.compile(pattern, re.IGNORECASE) for pattern in end_patterns]

    def matches_pattern(line: str, regexes: List[re.Pattern]) -> bool:
        return any(regex.search(line) for regex in regexes)

    chunks = []
    current_chunk_lines = []
    current_chunk_start_page = None
    current_chunk_pages = set()  # Track all pages this chunk spans

    for page in pages:
        for line in page_to_markdown(page, request).splitlines():
            # Check if this line starts a new chunk
            if start_regexes and matches_pattern(line, start_regexes):
                # Save previous chunk if it exists
                if current_chunk_lines:
                    content = "\n".join(current_chunk_lines).strip()
                    if content:
                        page_list = sorted(list(current_chunk_pages))
                        chunks.append(
                            Chunk(
                                content=content,
                                page_number=current_chunk_start_page,
                                page_numbers=page_list,
                            )
                        )

                # Start new chunk
                current_chunk_lines = [line]
                current_chunk_start_page = page.page_number
                current_chunk_pages = {page.page_number}

            # Add line to current chunk
            elif (
                current_chunk_lines or not start_regexes
            ):  # Include lines if we're in a chunk OR no start patterns
                if not current_chunk_lines:  # First line and no start patterns
                    current_chunk_start_page = page.page_number
                    current_chunk_pages = {page.page_number}
                else:
                    current_chunk_pages.add(page.page_number)
                current_chunk_lines.append(line)

            # Check if this line ends the current chunk
            if end_regexes and matches_pattern(line, end_regexes) and current_chunk_lines:
                content = "\n".join(current_chunk_lines).strip()
                if content:
                    page_list = sorted(list(current_chunk_pages))
                    chunks.append(
                        Chunk(
                            content=content,
                            page_number=current_chunk_start_page,
                            page_numbers=page_list,
                        )
                    )
                current_chunk_lines = []
                current_chunk_start_page = None
                current_chunk_pages = set()

    # Add final chunk if exists
    if current_chunk_lines:
        content = "\n".join(current_chunk_lines).strip()
        if content:
            page_list = sorted(list(current_chunk_pages))
            chunks.append(
                Chunk(content=content, page_number=current_chunk_start_page, page_numbers=page_list)
            )

    return chunks
