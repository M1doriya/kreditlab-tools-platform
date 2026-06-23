# SPDX-License-Identifier: Apache-2.0
"""Tests for AzureCellBboxExtractor — converts Azure table cells into
TextBoundingBox objects with pixel-space bounding boxes and stable ref ids.
"""

import pytest

from tensorlake_docai.ocr.azure_cell_bbox_extractor import AzureCellBboxExtractor


class _Region:
    def __init__(self, polygon):
        self.polygon = polygon


class _Cell:
    def __init__(self, content, polygon=None, row_index=None, col_index=None, snake_case=False):
        self.content = content
        if polygon is not None:
            region = _Region(polygon)
            if snake_case:
                self.bounding_regions = [region]
            else:
                self.boundingRegions = [region]
        # Allow both spellings of indices like Azure SDK can return.
        if row_index is not None:
            self.rowIndex = row_index
        if col_index is not None:
            self.columnIndex = col_index


class _Table:
    def __init__(self, cells):
        self.cells = cells


@pytest.fixture
def extractor():
    return AzureCellBboxExtractor()


# --- _polygon_to_bbox ----------------------------------------------------


def test_polygon_to_bbox_converts_inches_to_pixels(extractor):
    polygon = [1.0, 2.0, 3.0, 2.0, 3.0, 4.0, 1.0, 4.0]  # rectangle in inches
    bbox = extractor._polygon_to_bbox(polygon, scale_x=100, scale_y=200)
    assert bbox == {"x1": 100, "y1": 400, "x2": 300, "y2": 800}


def test_polygon_to_bbox_handles_short_polygon(extractor):
    assert extractor._polygon_to_bbox([], scale_x=1, scale_y=1) is None
    assert extractor._polygon_to_bbox([1, 2, 3], scale_x=1, scale_y=1) is None


# --- _get_cell_bbox ------------------------------------------------------


def test_get_cell_bbox_reads_camel_case_bounding_regions(extractor):
    cell = _Cell("hi", polygon=[1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0])
    bbox = extractor._get_cell_bbox(cell, scale_x=10, scale_y=10)
    assert bbox == {"x1": 10, "y1": 10, "x2": 20, "y2": 20}


def test_get_cell_bbox_reads_snake_case_bounding_regions(extractor):
    cell = _Cell(
        "hi",
        polygon=[1.0, 1.0, 2.0, 1.0, 2.0, 2.0, 1.0, 2.0],
        snake_case=True,
    )
    bbox = extractor._get_cell_bbox(cell, scale_x=10, scale_y=10)
    assert bbox == {"x1": 10, "y1": 10, "x2": 20, "y2": 20}


def test_get_cell_bbox_returns_none_without_polygon(extractor):
    cell = _Cell("hi")
    assert extractor._get_cell_bbox(cell, scale_x=10, scale_y=10) is None


# --- extract_table_cells_with_bboxes -------------------------------------


def test_extract_table_cells_returns_empty_for_no_cells(extractor):
    out = extractor.extract_table_cells_with_bboxes(
        _Table([]),
        page_width_pixels=612,
        page_height_pixels=792,
        page_width_inches=8.5,
        page_height_inches=11,
        page_number=1,
        reading_order=5,
    )
    assert out == []


def test_extract_table_cells_returns_empty_when_table_has_no_cells_attr(extractor):
    class NoCells:
        pass

    out = extractor.extract_table_cells_with_bboxes(
        NoCells(),
        page_width_pixels=612,
        page_height_pixels=792,
        page_width_inches=8.5,
        page_height_inches=11,
        page_number=1,
        reading_order=5,
    )
    assert out == []


def test_extract_table_cells_emits_bboxes_with_refs_and_indices(extractor):
    cells = [
        _Cell(
            "row0 col0",
            polygon=[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            row_index=0,
            col_index=0,
        ),
        _Cell(
            "row0 col1",
            polygon=[1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 1.0, 1.0],
            row_index=0,
            col_index=1,
        ),
    ]
    out = extractor.extract_table_cells_with_bboxes(
        _Table(cells),
        page_width_pixels=612,
        page_height_pixels=792,
        page_width_inches=8.5,
        page_height_inches=11,
        page_number=1,
        reading_order=5,
    )
    assert len(out) == 2
    assert out[0].text == "row0 col0"
    assert out[0].ref_id == "1.5.0"
    assert out[1].ref_id == "1.5.1"
    assert out[0].row_index == 0 and out[0].column_index == 0
    assert out[1].column_index == 1


def test_extract_table_cells_sorts_by_row_then_column(extractor):
    """Cells passed out of order must come back sorted by (row, col)."""
    cells = [
        _Cell(
            "r1c0",
            polygon=[0.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.0, 2.0],
            row_index=1,
            col_index=0,
        ),
        _Cell(
            "r0c1",
            polygon=[1.0, 0.0, 2.0, 0.0, 2.0, 1.0, 1.0, 1.0],
            row_index=0,
            col_index=1,
        ),
        _Cell(
            "r0c0",
            polygon=[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            row_index=0,
            col_index=0,
        ),
    ]
    out = extractor.extract_table_cells_with_bboxes(
        _Table(cells),
        page_width_pixels=600,
        page_height_pixels=600,
        page_width_inches=10,
        page_height_inches=10,
        page_number=2,
        reading_order=3,
    )
    assert [c.text for c in out] == ["r0c0", "r0c1", "r1c0"]
    # ref_ids reflect the sorted iteration order.
    assert [c.ref_id for c in out] == ["2.3.0", "2.3.1", "2.3.2"]


def test_extract_table_cells_skips_empty_content(extractor):
    cells = [
        _Cell(
            "",
            polygon=[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            row_index=0,
            col_index=0,
        ),
        _Cell(
            "real",
            polygon=[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            row_index=0,
            col_index=1,
        ),
    ]
    out = extractor.extract_table_cells_with_bboxes(
        _Table(cells),
        page_width_pixels=600,
        page_height_pixels=600,
        page_width_inches=10,
        page_height_inches=10,
        page_number=1,
        reading_order=0,
    )
    assert len(out) == 1
    assert out[0].text == "real"


def test_extract_table_cells_skips_cells_without_bbox(extractor):
    cells = [
        _Cell("no bbox", row_index=0, col_index=0),
        _Cell(
            "ok",
            polygon=[0.0, 0.0, 1.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            row_index=0,
            col_index=1,
        ),
    ]
    out = extractor.extract_table_cells_with_bboxes(
        _Table(cells),
        page_width_pixels=600,
        page_height_pixels=600,
        page_width_inches=10,
        page_height_inches=10,
        page_number=1,
        reading_order=0,
    )
    assert len(out) == 1
    assert out[0].text == "ok"
