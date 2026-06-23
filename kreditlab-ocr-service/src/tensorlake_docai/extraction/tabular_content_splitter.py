# SPDX-License-Identifier: Apache-2.0
"""
Tabular Content Splitter

Utilities for splitting dense tabular content (CSV/Excel/HTML tables) to handle
output token limits during structured extraction.
"""

from typing import List


def estimate_tokens(text: str, model: str = "gpt-4o") -> int:
    """Estimate token count for given text."""
    try:
        import tiktoken

        encoding = tiktoken.encoding_for_model(model)
        return len(encoding.encode(text))
    except Exception:
        # Fallback: roughly 4 characters per token for openai models
        return len(text) // 4


def is_table_row(line: str) -> bool:
    """Check if a line is a table row (markdown, CSV, or HTML format)."""
    stripped = line.strip()
    print(f"Is_table_row print out : Stripped: {stripped}")

    # Check for markdown table format
    if stripped.startswith("|") and stripped.endswith("|") and stripped.count("|") >= 2:
        return True

    # Check for CSV format (comma-separated with at least 2 columns)
    if "," in stripped and stripped.count(",") >= 1:
        return True

    # Check for HTML table row format
    if "<tr" in stripped.lower() and "</tr>" in stripped.lower():
        return True

    return False


def is_dense_tabular_content(content: str) -> bool:
    """Detect if content is dense tabular data (CSV/Excel-like or HTML tables)."""
    lines = content.strip().split("\n")
    print(f"Is_dense_tabular_content: Lines: {len(lines)}")

    # Check for HTML table first
    if "<table" in content.lower() and "</table>" in content.lower():
        try:
            from bs4 import BeautifulSoup

            soup = BeautifulSoup(content, "html.parser")
            table = soup.find("table")
            if table:
                rows = table.find_all("tr")
                if len(rows) >= 5:  # At least 5 rows
                    # Check column count from first row with data
                    for row in rows:
                        cells = row.find_all(["td", "th"])
                        if len(cells) >= 4:  # At least 4 columns
                            print(
                                f"Is_dense_tabular_content: HTML table with {len(rows)} rows, {len(cells)} columns"
                            )
                            return True
        except Exception as e:
            print(f"Error parsing HTML table: {e}")
            # Fall through to line-based detection

    if len(lines) < 3:  # Need at least header + separator + data
        return False

    # Count table rows (markdown/CSV format)
    table_rows = [line for line in lines if is_table_row(line)]
    print(f"Is_dense_tabular_content: Table rows: {len(table_rows)}")

    # Consider it dense tabular if:
    # 1. More than 70% of lines are table rows
    # 2. At least 5 table rows
    # 3. Table rows have many columns (>= 4)
    if len(table_rows) >= 5 and len(table_rows) / len(lines) > 0.7:
        # Check if rows have many columns
        sample_row = table_rows[0] if table_rows else ""

        # Count columns for different formats
        if sample_row.startswith("|") and sample_row.endswith("|"):
            # Markdown table format
            column_count = sample_row.count("|") - 1
        else:
            # CSV format
            column_count = sample_row.count(",") + 1

        print(f"Is_dense_tabular_content: Column count: {column_count}")
        return column_count >= 4

    return False


def should_split_dense_table(schema: dict, content: str, model: str = "gpt-4o") -> bool:
    """Check if dense tabular content should be chunked based on schema complexity."""
    if not is_dense_tabular_content(content):
        return False

    # Count table rows
    lines = content.strip().split("\n")
    table_rows = [
        line
        for line in lines
        if is_table_row(line)
        and not (line.strip().replace("|", "").replace("-", "").replace(" ", "") == "")
    ]

    print(f"Table rows: {len(table_rows)}")

    if len(table_rows) < 3:  # Need meaningful number of rows
        return False

    # Estimate schema tokens
    schema_tokens = estimate_tokens(str(schema), model)
    print(f"Schema tokens: {schema_tokens}")

    # Use threshold of 12000 instead of 7000 for better efficiency
    # Account for prompt overhead (~2000 tokens)
    threshold = 10000

    # Check if schema_tokens * row_count exceeds threshold
    estimated_output = schema_tokens * len(table_rows)
    print(f"Estimated output: {estimated_output}")

    return estimated_output > threshold


def split_per_table(content: str) -> List[str]:
    """Split content into chunks where each table becomes its own chunk."""
    lines = content.strip().split("\n")
    chunks = []
    current_chunk = []
    in_table = False

    for line in lines:
        is_table_line = is_table_row(line)

        if is_table_line:
            if not in_table:
                # Starting a new table - finish previous chunk if exists
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                in_table = True
            current_chunk.append(line)
        else:
            if in_table:
                # End of table - finish this table chunk
                if current_chunk:
                    chunks.append("\n".join(current_chunk))
                    current_chunk = []
                in_table = False
            current_chunk.append(line)

    # Add final chunk if exists
    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks if chunks else [content]


def split_dense_table_content(
    content: str, max_tokens_per_chunk: int, schema_tokens: int
) -> List[str]:
    """Split dense tabular content into chunks with headers preserved."""
    # Check if this is HTML table content
    if "<table" in content.lower() and "</table>" in content.lower():
        return split_html_table_content(content, max_tokens_per_chunk, schema_tokens)

    # Handle markdown/CSV content (existing logic)
    lines = content.strip().split("\n")
    print("Table rows: ", len(lines))

    # Check for multiple tables - if detected, fall back to simple text chunking
    table_headers_count = 0
    for i, line in enumerate(lines):
        if is_table_row(line):
            # Check if this could be a new table header (after non-table content or another complete table)
            if i == 0 or not is_table_row(lines[i - 1]):
                table_headers_count += 1
                if table_headers_count > 1:
                    print("Multiple tables detected, chunking per table")
                    return split_per_table(content)

    # Find header and data rows
    header_lines = []
    data_rows = []

    i = 0
    while i < len(lines):
        line = lines[i]
        if is_table_row(line):
            if not header_lines:
                header_lines.append(line)
                # Check for separator row
                if (
                    i + 1 < len(lines)
                    and lines[i + 1].strip().replace("|", "").replace("-", "").replace(" ", "")
                    == ""
                ):
                    header_lines.append(lines[i + 1])
                    i += 1  # Skip the separator row
            elif (
                len(header_lines) == 1
                and line.strip().replace("|", "").replace("-", "").replace(" ", "") == ""
            ):
                header_lines.append(line)  # Separator row
            else:
                data_rows.append(line)
        i += 1

    if not header_lines or not data_rows:
        return [content]  # Not a proper table, return as-is

    header_content = "\n".join(header_lines)

    # Split into chunks with n rows in each chunks that will fit in the estimated output tokens
    # determine n
    n = max(1, int(max_tokens_per_chunk / schema_tokens)) if schema_tokens > 0 else 1
    print(f"Number of rows per chunk: {n}")

    chunks = []
    for i in range(0, len(data_rows), n):
        chunk_data_rows = data_rows[i : i + n]
        chunk_content = header_content + "\n" + "\n".join(chunk_data_rows)
        chunks.append(chunk_content)

    return chunks if chunks else [content]


def split_html_table_content(
    content: str, max_tokens_per_chunk: int, schema_tokens: int
) -> List[str]:
    """Split HTML table content into chunks with headers preserved."""
    try:
        from bs4 import BeautifulSoup
        import copy

        soup = BeautifulSoup(content, "html.parser")
        table = soup.find("table")
        if not table:
            return [content]

        all_rows = table.find_all("tr")
        print(f"HTML table rows: {len(all_rows)}")

        if len(all_rows) < 3:  # Need meaningful number of rows
            return [content]

        # Find header rows (first few rows, typically with th tags or first tr)
        header_rows = []
        data_rows = []

        for i, row in enumerate(all_rows):
            # Consider first row or rows with th tags as headers
            if i == 0 or row.find_all("th"):
                header_rows.append(row)
            else:
                data_rows.append(row)

        if not data_rows:
            return [content]  # No data rows found

        # Calculate number of rows per chunk
        n = max(1, int(max_tokens_per_chunk / schema_tokens)) if schema_tokens > 0 else 1
        print(f"Number of HTML rows per chunk: {n}")

        # Create chunks
        chunks = []
        table_class = table.get("class", [])

        for i in range(0, len(data_rows), n):
            chunk_data_rows = data_rows[i : i + n]

            # Create new table for this chunk
            chunk_table = BeautifulSoup("<table></table>", "html.parser").find("table")
            if table_class:
                chunk_table["class"] = table_class

            # Add header rows
            for header_row in header_rows:
                chunk_table.append(copy.deepcopy(header_row))

            # Add data rows for this chunk
            for data_row in chunk_data_rows:
                chunk_table.append(copy.deepcopy(data_row))

            chunks.append(str(chunk_table))

        print(f"Created {len(chunks)} HTML table chunks")
        return chunks if chunks else [content]

    except Exception as e:
        print(f"Error chunking HTML table: {e}")
        return [content]  # Fall back to original content


def merge_extraction_results(results: List[dict], schema: dict) -> dict:
    """Merge results from multiple chunks into a single result."""
    if not results:
        return {}

    if len(results) == 1:
        return results[0]

    # For tabular data (CSV/Excel), simply collect all results into arrays
    merged = {}

    for key in results[0].keys():
        values = [result.get(key) for result in results if result.get(key) is not None]

        if not values:
            merged[key] = []
            continue

        # Collect all values into arrays
        merged_list = []
        for value in values:
            if isinstance(value, list):
                merged_list.extend(value)
            else:
                merged_list.append(value)
        merged[key] = merged_list

    return merged


def is_output_token_limit_error(error_message: str) -> bool:
    """Check if an error message indicates an output token limit was reached."""
    return any(
        pattern in error_message
        for pattern in [
            "CompletionUsage(completion_tokens=16384",
            "maximum completion tokens",
            "output token limit",
            "completion_tokens=16384",
        ]
    )
