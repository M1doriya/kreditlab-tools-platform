# SPDX-License-Identifier: Apache-2.0
"""HTML/string utilities used during cross-page table merging."""

from tensorlake_docai.tables.table_merging import (
    _is_vertically_aligned,
    are_tables_semantically_aligned,
    extract_json_from_response,
    get_cross_page_table_candidates,
    get_cross_page_tables,
    get_first_table,
    get_last_table,
    get_table_column_count,
    get_table_column_types,
    get_tables_from_page_start,
    infer_cell_type,
    merge_table_htmls,
    remove_header_rows_regex,
    slice_table_rows,
)
from tensorlake_docai.pipeline.api import (
    Page,
    PageFragment,
    PageFragmentType,
    Table,
    Text,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_table_fragment(html: str = "<table></table>", bbox=None) -> PageFragment:
    return PageFragment(
        fragment_type=PageFragmentType.TABLE,
        content=Table(content="", cells=[], html=html),
        bbox=bbox,
    )


def _make_text_fragment() -> PageFragment:
    return PageFragment(
        fragment_type=PageFragmentType.TEXT,
        content=Text(content="Some text"),
    )


def _make_page(number: int, fragments=None) -> Page:
    return Page(page_number=number, page_fragments=fragments or [])


# --- merge_table_htmls ----------------------------------------------------


def test_merge_table_htmls_appends_rows():
    base = "<table><tr><td>a</td></tr></table>"
    nxt = "<table><tr><td>b</td></tr></table>"
    merged = merge_table_htmls(base, nxt)
    assert merged.count("<tr>") == 2
    assert "<td>a</td>" in merged and "<td>b</td>" in merged
    assert merged.strip().endswith("</table>")


def test_merge_table_htmls_skips_header_rows():
    base = "<table><tr><td>keep</td></tr></table>"
    # Two rows in next; skip the first.
    nxt = "<table><tr><td>drop</td></tr><tr><td>add</td></tr></table>"
    merged = merge_table_htmls(base, nxt, skip_rows=1)
    assert "<td>drop</td>" not in merged
    assert "<td>add</td>" in merged
    assert "<td>keep</td>" in merged


def test_merge_table_htmls_empty_next_returns_base():
    base = "<table><tr><td>a</td></tr></table>"
    assert merge_table_htmls(base, "<table></table>") == base


# --- slice_table_rows -----------------------------------------------------


def test_slice_table_rows_first_two():
    html = "<table><tr><td>1</td></tr>" "<tr><td>2</td></tr>" "<tr><td>3</td></tr></table>"
    out = slice_table_rows(html, end=2)
    assert "<td>1</td>" in out and "<td>2</td>" in out
    assert "<td>3</td>" not in out


def test_slice_table_rows_preserves_open_tag_attrs():
    html = '<table border="1"><tr><td>x</td></tr></table>'
    out = slice_table_rows(html, end=1)
    assert '<table border="1">' in out


# --- remove_header_rows_regex ---------------------------------------------


def test_remove_header_rows_regex_strips_n_rows():
    body = "<tr><td>a</td></tr><tr><td>b</td></tr><tr><td>c</td></tr>"
    assert remove_header_rows_regex(body, 0) == body
    out = remove_header_rows_regex(body, 2)
    assert "<td>a</td>" not in out
    assert "<td>b</td>" not in out
    assert "<td>c</td>" in out


# --- extract_json_from_response -------------------------------------------


def test_extract_json_plain():
    assert extract_json_from_response('{"a": 1}') == {"a": 1}


def test_extract_json_code_fenced():
    text = '```json\n{"a": 1, "b": [2]}\n```'
    assert extract_json_from_response(text) == {"a": 1, "b": [2]}


def test_extract_json_embedded_in_prose():
    text = 'Here you go: {"a": 1} — done.'
    assert extract_json_from_response(text) == {"a": 1}


def test_extract_json_invalid_returns_empty_dict():
    assert extract_json_from_response("nothing structured here") == {}


# --- column count / cell type ---------------------------------------------


def test_get_table_column_count_simple():
    html = "<table><tr><td>1</td><td>2</td><td>3</td></tr></table>"
    assert get_table_column_count(html) == 3


def test_get_table_column_count_with_colspan():
    html = '<table><tr><td colspan="2">x</td><td>y</td></tr></table>'
    assert get_table_column_count(html) == 3


def test_get_table_column_count_empty():
    assert get_table_column_count("<table></table>") == 0


def test_infer_cell_type():
    assert infer_cell_type("") == "empty"
    assert infer_cell_type("   ") == "empty"
    assert infer_cell_type("123") == "number"
    assert infer_cell_type("$1,234.50") == "number"
    assert infer_cell_type("(1,234)") == "number"
    assert infer_cell_type("42%") == "number"
    assert infer_cell_type("Alice") == "text"


def test_get_table_column_types_counts_headers_as_text():
    html = (
        "<table>"
        "<tr><td>Name</td><td>Age</td><td>Salary</td></tr>"
        "<tr><td>Alice</td><td>30</td><td>$50,000</td></tr>"
        "<tr><td>Bob</td><td>25</td><td>$60,000</td></tr>"
        "</table>"
    )
    types = get_table_column_types(html, rows_to_check=3, from_start=True)
    assert types == ["text", "text", "text"]


def test_get_table_column_types_detects_numeric_data_without_headers():
    html = (
        "<table>"
        "<tr><td>Alice</td><td>30</td><td>$50,000</td></tr>"
        "<tr><td>Bob</td><td>25</td><td>$60,000</td></tr>"
        "</table>"
    )
    assert get_table_column_types(html, rows_to_check=2, from_start=True) == [
        "text",
        "number",
        "number",
    ]


def test_get_table_column_types_from_end():
    html = (
        "<table>"
        "<tr><td>Name</td><td>Score</td></tr>"
        "<tr><td>Alice</td><td>95</td></tr>"
        "<tr><td>Bob</td><td>88</td></tr>"
        "</table>"
    )
    types = get_table_column_types(html, rows_to_check=2, from_start=False)
    assert types == ["text", "number"]


def test_get_table_column_types_empty_table():
    assert get_table_column_types("<table></table>") == []


# --- get_last_table / get_first_table ------------------------------------


def test_get_last_table_returns_last():
    t1 = _make_table_fragment("<table>1</table>")
    t2 = _make_table_fragment("<table>2</table>")
    txt = _make_text_fragment()
    fragments = [t1, txt, t2]
    result = get_last_table(fragments)
    assert result is t2


def test_get_last_table_no_table_returns_none():
    assert get_last_table([_make_text_fragment()]) is None


def test_get_last_table_empty_list_returns_none():
    assert get_last_table([]) is None


def test_get_first_table_returns_first():
    t1 = _make_table_fragment("<table>1</table>")
    t2 = _make_table_fragment("<table>2</table>")
    result = get_first_table([_make_text_fragment(), t1, t2])
    assert result is t1


def test_get_first_table_no_table_returns_none():
    assert get_first_table([_make_text_fragment()]) is None


# --- get_cross_page_tables -----------------------------------------------


def test_get_cross_page_tables_finds_pair():
    t_end = _make_table_fragment("<table>end</table>")
    t_start = _make_table_fragment("<table>start</table>")
    pages = [
        _make_page(1, [_make_text_fragment(), t_end]),
        _make_page(2, [t_start, _make_text_fragment()]),
    ]
    end, start = get_cross_page_tables(pages, 0)
    assert end is t_end
    assert start is t_start


def test_get_cross_page_tables_last_page_returns_none_pair():
    pages = [_make_page(1, [_make_table_fragment()])]
    end, start = get_cross_page_tables(pages, 0)
    assert end is None
    assert start is None


def test_get_cross_page_tables_no_table_on_either_side():
    pages = [
        _make_page(1, [_make_text_fragment()]),
        _make_page(2, [_make_text_fragment()]),
    ]
    end, start = get_cross_page_tables(pages, 0)
    assert end is None
    assert start is None


# --- get_cross_page_table_candidates ------------------------------------


def test_get_cross_page_table_candidates_returns_pairs():
    t1 = _make_table_fragment()
    t2 = _make_table_fragment()
    pages = [
        _make_page(1, [t1]),
        _make_page(2, [t2]),
    ]
    candidates = get_cross_page_table_candidates(pages, 0)
    assert len(candidates) == 1
    assert candidates[0] == (t1, t2)


def test_get_cross_page_table_candidates_last_page_returns_empty():
    pages = [_make_page(1, [_make_table_fragment()])]
    assert get_cross_page_table_candidates(pages, 0) == []


# --- get_tables_from_page_start -----------------------------------------


def test_get_tables_from_page_start_respects_limit():
    frags = [_make_table_fragment() for _ in range(5)]
    page = _make_page(1, frags)
    result = get_tables_from_page_start(page, limit=3)
    assert len(result) == 3


def test_get_tables_from_page_start_skips_non_tables():
    page = _make_page(1, [_make_text_fragment(), _make_table_fragment(), _make_text_fragment()])
    result = get_tables_from_page_start(page, limit=5)
    assert len(result) == 1


# --- _is_vertically_aligned ---------------------------------------------


def test_is_vertically_aligned_full_overlap():
    b = {"x1": 0, "y1": 0, "x2": 100, "y2": 50}
    f1 = _make_table_fragment(bbox=b)
    f2 = _make_table_fragment(bbox=b)
    assert _is_vertically_aligned(f1, f2)


def test_is_vertically_aligned_no_overlap():
    f1 = _make_table_fragment(bbox={"x1": 0, "y1": 0, "x2": 50, "y2": 10})
    f2 = _make_table_fragment(bbox={"x1": 200, "y1": 20, "x2": 300, "y2": 30})
    assert not _is_vertically_aligned(f1, f2)


def test_is_vertically_aligned_no_bbox_returns_true():
    f1 = _make_table_fragment()
    f2 = _make_table_fragment()
    assert _is_vertically_aligned(f1, f2)


# --- are_tables_semantically_aligned ------------------------------------


def test_are_tables_semantically_aligned_matching_types():
    html = (
        "<table>"
        "<tr><td>Alice</td><td>100</td></tr>"
        "<tr><td>Bob</td><td>200</td></tr>"
        "</table>"
    )
    assert are_tables_semantically_aligned(html, html)


def test_are_tables_semantically_aligned_different_column_counts():
    html1 = "<table><tr><td>A</td><td>1</td></tr></table>"
    html2 = "<table><tr><td>A</td><td>1</td><td>extra</td></tr></table>"
    assert not are_tables_semantically_aligned(html1, html2)


def test_are_tables_semantically_aligned_empty_returns_true():
    assert are_tables_semantically_aligned("<table></table>", "<table></table>")
