# SPDX-License-Identifier: Apache-2.0
"""Dense-table splitter — the chunking math here decides whether a structured
extraction call gets truncated by the output-token limit."""

from tensorlake_docai.extraction.tabular_content_splitter import (
    estimate_tokens,
    is_dense_tabular_content,
    is_output_token_limit_error,
    is_table_row,
    merge_extraction_results,
    should_split_dense_table,
    split_dense_table_content,
    split_per_table,
)

# --- row detection ---------------------------------------------------------


def test_is_table_row_markdown():
    assert is_table_row("| a | b | c |")
    assert is_table_row("|---|---|---|")


def test_is_table_row_csv():
    assert is_table_row("a,b,c")


def test_is_table_row_html():
    assert is_table_row("<tr><td>x</td></tr>")
    assert is_table_row("<TR><TD>x</TD></TR>")  # case insensitive


def test_is_table_row_plain_text_rejected():
    assert not is_table_row("just some prose")
    assert not is_table_row("")


# --- density detection ----------------------------------------------------


def _markdown_table(rows: int, cols: int = 5) -> str:
    header = "| " + " | ".join(f"c{i}" for i in range(cols)) + " |"
    sep = "|" + "|".join(["---"] * cols) + "|"
    body = ["| " + " | ".join(f"v{i}{j}" for j in range(cols)) + " |" for i in range(rows)]
    return "\n".join([header, sep, *body])


def test_is_dense_tabular_content_markdown_table():
    assert is_dense_tabular_content(_markdown_table(rows=10, cols=5))


def test_is_dense_tabular_content_short_table_rejected():
    # 3 data rows + header + sep = 5 rows total; less rows = not dense
    assert not is_dense_tabular_content(_markdown_table(rows=2, cols=5))


def test_is_dense_tabular_content_narrow_table_rejected():
    # 10 rows but only 2 columns — should not be considered dense
    assert not is_dense_tabular_content(_markdown_table(rows=10, cols=2))


def test_is_dense_tabular_content_html_table():
    rows_html = "".join(
        f"<tr><td>a{i}</td><td>b{i}</td><td>c{i}</td><td>d{i}</td></tr>" for i in range(6)
    )
    content = f"<table>{rows_html}</table>"
    assert is_dense_tabular_content(content)


# --- should_split_dense_table ---------------------------------------------


def test_should_split_dense_table_small_schema_no_split():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    # Small schema * 10 rows is below the 10k threshold
    assert not should_split_dense_table(schema, _markdown_table(rows=10, cols=5))


def test_should_split_dense_table_huge_schema_splits():
    # Force a schema large enough that schema_tokens * rows exceeds threshold
    big_descr = "x" * 50000  # ~12500 tokens fallback
    schema = {"type": "object", "properties": {"a": {"type": "string", "description": big_descr}}}
    assert should_split_dense_table(schema, _markdown_table(rows=10, cols=5))


def test_should_split_dense_table_non_table_content():
    assert not should_split_dense_table({"type": "object"}, "just prose, nothing tabular here.")


# --- splitting ------------------------------------------------------------


def test_split_dense_table_content_preserves_header_per_chunk():
    content = _markdown_table(rows=10, cols=4)
    # max_tokens_per_chunk / schema_tokens => 3 rows per chunk
    chunks = split_dense_table_content(content, max_tokens_per_chunk=30, schema_tokens=10)
    assert len(chunks) >= 2
    header = "| c0 | c1 | c2 | c3 |"
    sep = "|---|---|---|---|"
    for chunk in chunks:
        assert header in chunk
        assert sep in chunk


def test_split_per_table_isolates_tables():
    content = (
        "intro line\n"
        "| a | b |\n|---|---|\n| 1 | 2 |\n"
        "between line\n"
        "| x | y |\n|---|---|\n| 9 | 8 |\n"
        "outro line"
    )
    chunks = split_per_table(content)
    # At minimum: prose-before, first-table, between, second-table, prose-after
    table_chunks = [c for c in chunks if "|" in c]
    assert len(table_chunks) == 2


# --- merge_extraction_results --------------------------------------------


def test_merge_extraction_results_empty():
    assert merge_extraction_results([], {}) == {}


def test_merge_extraction_results_passthrough_single():
    only = {"rows": [{"a": 1}]}
    assert merge_extraction_results([only], {}) == only


def test_merge_extraction_results_concatenates_lists():
    results = [{"rows": [{"a": 1}]}, {"rows": [{"a": 2}, {"a": 3}]}]
    merged = merge_extraction_results(results, {})
    assert merged["rows"] == [{"a": 1}, {"a": 2}, {"a": 3}]


def test_merge_extraction_results_drops_missing_keys():
    results = [{"rows": [{"a": 1}]}, {"rows": None}]
    merged = merge_extraction_results(results, {})
    assert merged["rows"] == [{"a": 1}]


# --- token-limit detection ------------------------------------------------


def test_is_output_token_limit_error_patterns():
    assert is_output_token_limit_error("hit maximum completion tokens")
    assert is_output_token_limit_error("hit the output token limit")
    assert is_output_token_limit_error("CompletionUsage(completion_tokens=16384, ...)")
    assert not is_output_token_limit_error("network error")
    assert not is_output_token_limit_error("")


def test_estimate_tokens_returns_positive_int():
    n = estimate_tokens("hello world", model="gpt-4o")
    assert isinstance(n, int) and n > 0
