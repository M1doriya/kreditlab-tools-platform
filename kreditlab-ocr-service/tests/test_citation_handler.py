# SPDX-License-Identifier: Apache-2.0
"""Tests for citation_handler — ref-id injection and citation resolution.

Covers the pure logic: HTML cell ref injection, citation map population
from layout objects, _ref → _citation rewriting, page/ref filtering,
schema enhancement for citations.
"""

import pytest

from tensorlake_docai.extraction.citation_handler import (
    StructuredExtractionCitationHandler,
)
from tensorlake_docai.models.layout_objects import PageLayout, PageLayoutElement, TextBoundingBox
from tensorlake_docai.pipeline.api import PageFragmentType


@pytest.fixture
def handler():
    return StructuredExtractionCitationHandler()


# --- _bbox_tuple_to_dict --------------------------------------------------


def test_bbox_tuple_to_dict_basic(handler):
    out = handler._bbox_tuple_to_dict((1.0, 2.0, 3.0, 4.0))
    assert out == {"x1": 1.0, "y1": 2.0, "x2": 3.0, "y2": 4.0}


def test_bbox_tuple_to_dict_with_page_number(handler):
    out = handler._bbox_tuple_to_dict((1, 2, 3, 4), page_number=5)
    assert out["page_number"] == 5


def test_bbox_tuple_to_dict_rejects_bad_shape(handler):
    assert handler._bbox_tuple_to_dict(None) is None
    assert handler._bbox_tuple_to_dict((1, 2)) is None
    assert handler._bbox_tuple_to_dict(()) is None


# --- enhance_html_with_cell_refs -----------------------------------------


def _cell_bbox(row, col, ref):
    return TextBoundingBox(
        bbox=(0.0, 0.0, 1.0, 1.0), text="x", ref_id=ref, row_index=row, column_index=col
    )


def test_enhance_html_with_cell_refs_injects_refs(handler):
    html = "<table>" "<tr><th>A</th><th>B</th></tr>" "<tr><td>1</td><td>2</td></tr>" "</table>"
    bboxes = [
        _cell_bbox(0, 0, "1.5.0"),
        _cell_bbox(1, 1, "1.5.3"),
    ]
    out = handler.enhance_html_with_cell_refs(html, bboxes)
    assert "[REF:1.5.0]" in out
    assert "[REF:1.5.3]" in out
    # Cells without a mapped ref aren't tagged.
    # (row 0 col 1 → "B" — no [REF: appended right after the B)
    assert "B [REF:" not in out


def test_enhance_html_empty_inputs_pass_through(handler):
    assert handler.enhance_html_with_cell_refs("", []) == ""
    assert handler.enhance_html_with_cell_refs("<p>x</p>", []) == "<p>x</p>"


def test_enhance_html_skips_bboxes_missing_position(handler):
    html = "<table><tr><td>1</td></tr></table>"
    # No row/col indices → ignored
    bbox = TextBoundingBox(bbox=(0.0, 0.0, 1.0, 1.0), text="x", ref_id="r", row_index=None)
    out = handler.enhance_html_with_cell_refs(html, [bbox])
    assert out == html


# --- populate_citation_map -----------------------------------------------


def _layout(page_number, elements):
    return PageLayout(elements=elements, shape=(100, 100), page_number=page_number)


def _elem(ftype, ref_id, reading_order, bbox=(0.0, 0.0, 1.0, 1.0), cells=None):
    return PageLayoutElement(
        bbox=bbox,
        fragment_type=ftype,
        score=0.9,
        reading_order=reading_order,
        ref_id=ref_id,
        text_bounding_boxes=cells,
        ocr_text="x",
    )


def test_populate_citation_map_adds_explicit_and_fallback_refs(handler):
    layout = _layout(
        1,
        [_elem(PageFragmentType.TEXT, ref_id="1.0", reading_order=0)],
    )
    handler.populate_citation_map([layout])

    assert "1.0" in handler.citation_map
    bbox = handler.citation_map["1.0"]
    assert bbox["page_number"] == 1
    # Fallback ref `{page}.{reading_order}` is also indexed for the same element.
    assert "1.0" in handler.citation_map


def test_populate_citation_map_handles_cell_refs(handler):
    cells = [_cell_bbox(0, 0, "1.5.0"), _cell_bbox(0, 1, "1.5.1")]
    layout = _layout(
        1,
        [_elem(PageFragmentType.TABLE, ref_id="1.5", reading_order=5, cells=cells)],
    )
    handler.populate_citation_map([layout])
    assert "1.5.0" in handler.citation_map
    assert "1.5.1" in handler.citation_map


def test_populate_citation_map_handles_empty(handler):
    handler.populate_citation_map([])
    assert handler.citation_map == {}


# --- prepare_text_with_citations -----------------------------------------


def test_prepare_text_emits_refs_for_text_and_headers(handler):
    layout = _layout(
        2,
        [
            PageLayoutElement(
                bbox=(0, 0, 1, 1),
                fragment_type=PageFragmentType.SECTION_HEADER,
                score=0.9,
                reading_order=0,
                ref_id="2.0",
                ocr_text="Heading",
            ),
            PageLayoutElement(
                bbox=(0, 0, 1, 1),
                fragment_type=PageFragmentType.TEXT,
                score=0.9,
                reading_order=1,
                ref_id="2.1",
                ocr_text="Body text",
            ),
        ],
    )
    text, has = handler.prepare_text_with_citations("ignored", [layout])
    assert has is True
    assert "[REF:2.0:HEADER] Heading" in text
    assert "[REF:2.1] Body text" in text
    assert "--- Page 2 ---" in text


def test_prepare_text_skips_empty_ocr(handler):
    layout = _layout(
        1,
        [
            PageLayoutElement(
                bbox=(0, 0, 1, 1),
                fragment_type=PageFragmentType.TEXT,
                score=0.9,
                reading_order=0,
                ref_id="1.0",
                ocr_text="   ",
            ),
        ],
    )
    text, has = handler.prepare_text_with_citations("", [layout])
    assert has is False
    assert "[REF:" not in text


# --- enhance_schema_for_citations ----------------------------------------


def test_enhance_schema_adds_ref_for_simple_fields(handler):
    schema = {
        "type": "object",
        "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
    }
    out = handler.enhance_schema_for_citations(schema)
    assert "name_ref" in out["properties"]
    assert "age_ref" in out["properties"]
    assert out["properties"]["name_ref"]["type"] == "array"
    assert set(out["required"]) >= {"name_ref", "age_ref"}


def test_enhance_schema_recurses_into_nested_objects(handler):
    schema = {
        "type": "object",
        "properties": {
            "inner": {
                "type": "object",
                "properties": {"x": {"type": "string"}},
            }
        },
    }
    out = handler.enhance_schema_for_citations(schema)
    # No `inner_ref` at the outer level — nested objects get refs on their leaves instead.
    assert "inner_ref" not in out["properties"]
    assert "x_ref" in out["properties"]["inner"]["properties"]


def test_enhance_schema_handles_array_of_scalars(handler):
    schema = {
        "type": "object",
        "properties": {"tags": {"type": "array", "items": {"type": "string"}}},
    }
    out = handler.enhance_schema_for_citations(schema)
    assert "tags_ref" in out["properties"]


def test_enhance_schema_handles_array_of_objects(handler):
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {"type": "object", "properties": {"name": {"type": "string"}}},
            }
        },
    }
    out = handler.enhance_schema_for_citations(schema)
    # `items_ref` is NOT added; refs are added to leaves of the items object.
    assert "items_ref" not in out["properties"]
    assert "name_ref" in out["properties"]["items"]["items"]["properties"]


# --- add_citation_instructions -------------------------------------------


def test_add_citation_instructions_appends_to_user_prompt(handler):
    out = handler.add_citation_instructions("Extract everything.")
    assert out.startswith("Extract everything.")
    assert "_ref" in out
    assert "[REF:" in out


def test_add_citation_instructions_handles_none(handler):
    out = handler.add_citation_instructions(None)
    assert "_ref" in out


# --- resolve_citations ----------------------------------------------------


def test_resolve_citations_maps_single_ref(handler):
    handler.citation_map = {"1.0": {"x1": 1, "y1": 2, "x2": 3, "y2": 4, "page_number": 1}}
    data = {"name": "Acme", "name_ref": "1.0"}
    out = handler.resolve_citations(data)
    assert out["name"] == "Acme"
    assert out["name_citation"]["x1"] == 1
    # _ref field itself is not copied verbatim.
    assert "name_ref" not in out


def test_resolve_citations_maps_list_refs(handler):
    handler.citation_map = {
        "1.0": {"x1": 0, "y1": 0, "x2": 1, "y2": 1, "page_number": 1},
        "1.1": {"x1": 2, "y1": 2, "x2": 3, "y2": 3, "page_number": 1},
    }
    data = {"name": "x", "name_ref": ["1.0", "1.1", "missing"]}
    out = handler.resolve_citations(data)
    # Missing refs are filtered out of the list.
    assert len(out["name_citation"]) == 2


def test_resolve_citations_returns_empty_list_for_unmapped_single_ref(handler):
    handler.citation_map = {}
    data = {"name": "x", "name_ref": "missing"}
    out = handler.resolve_citations(data)
    assert out["name_citation"] == []


def test_resolve_citations_allowed_pages_filter(handler):
    handler.citation_map = {
        "1.0": {"x1": 0, "y1": 0, "x2": 1, "y2": 1, "page_number": 1},
        "2.0": {"x1": 0, "y1": 0, "x2": 1, "y2": 1, "page_number": 2},
    }
    data = {"name": "x", "name_ref": ["1.0", "2.0"]}
    out = handler.resolve_citations(data, allowed_pages={1})
    assert len(out["name_citation"]) == 1
    assert out["name_citation"][0]["page_number"] == 1


def test_resolve_citations_allowed_ref_ids_filter(handler):
    handler.citation_map = {
        "1.0": {"x1": 0, "y1": 0, "x2": 1, "y2": 1, "page_number": 1},
        "1.1": {"x1": 0, "y1": 0, "x2": 1, "y2": 1, "page_number": 1},
    }
    data = {"name": "x", "name_ref": ["1.0", "1.1"]}
    out = handler.resolve_citations(data, allowed_ref_ids={"1.0"})
    assert len(out["name_citation"]) == 1


def test_resolve_citations_recurses_into_nested(handler):
    handler.citation_map = {"1.0": {"x1": 0, "y1": 0, "x2": 1, "y2": 1, "page_number": 1}}
    data = {
        "outer": {"inner": "value", "inner_ref": "1.0"},
        "lst": [{"k": "v", "k_ref": "1.0"}],
    }
    out = handler.resolve_citations(data)
    assert out["outer"]["inner_citation"]["x1"] == 0
    assert out["lst"][0]["k_citation"]["x1"] == 0


def test_resolve_citations_with_none_ref_keeps_original_only(handler):
    data = {"name": "x", "name_ref": None}
    out = handler.resolve_citations(data)
    assert out == {"name": "x"}


def test_resolve_citations_non_dict_returns_as_is(handler):
    assert handler.resolve_citations("not a dict") == "not a dict"
    assert handler.resolve_citations([1, 2]) == [1, 2]
