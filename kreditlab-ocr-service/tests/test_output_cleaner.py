# SPDX-License-Identifier: Apache-2.0
"""Tests for OutputCleaner — regex-based dots.mocr output salvage logic.

This module is responsible for taking raw (often malformed) model output and
turning it into a list of well-formed layout dicts. Bugs here directly
corrupt downstream layout, so it warrants tight unit coverage.
"""

import json

import pytest

from tensorlake_docai.postprocess.output_cleaner import OutputCleaner, CleanedData


@pytest.fixture
def cleaner():
    return OutputCleaner()


# --- clean_list_data -------------------------------------------------------


def test_clean_list_keeps_well_formed_items(cleaner):
    data = [
        {"bbox": [0, 0, 10, 10], "category": "Text", "text": "hello"},
        {"bbox": [5, 5, 20, 20], "category": "Title", "text": "x"},
    ]
    result = cleaner.clean_list_data(data, case_id=1)

    assert result.success is True
    assert result.cleaned_data == data
    assert result.cleaning_operations["final_count"] == 2
    assert result.cleaning_operations["bbox_fixes"] == 0
    assert result.cleaning_operations["removed_items"] == 0


def test_clean_list_fixes_3_coord_bbox(cleaner):
    """bbox with 3 coords is salvaged — drop bbox, keep category+text."""
    data = [{"bbox": [0, 0, 10], "category": "Text", "text": "salvaged"}]
    result = cleaner.clean_list_data(data, case_id=1)

    assert result.cleaned_data == [{"category": "Text", "text": "salvaged"}]
    assert result.cleaning_operations["bbox_fixes"] == 1


def test_clean_list_drops_3_coord_bbox_with_no_content(cleaner):
    """3-coord bbox with no category and no text is fully dropped."""
    data = [{"bbox": [0, 0, 10]}]
    result = cleaner.clean_list_data(data, case_id=1)

    assert result.cleaned_data == []
    assert result.cleaning_operations["removed_items"] == 1


def test_clean_list_drops_non_dicts_and_garbage_bbox(cleaner):
    data = [
        "not a dict",
        {"bbox": "garbage", "category": "Text"},
        {"bbox": [1, 2, 3, 4, 5], "category": "Text"},
    ]
    result = cleaner.clean_list_data(data, case_id=1)

    assert result.cleaned_data == []
    assert result.cleaning_operations["removed_items"] == 3


def test_clean_list_keeps_no_bbox_with_category(cleaner):
    data = [{"category": "Text", "text": "no bbox is fine"}]
    result = cleaner.clean_list_data(data, case_id=1)

    assert len(result.cleaned_data) == 1
    assert result.cleaned_data[0]["category"] == "Text"


def test_clean_list_drops_no_bbox_no_category(cleaner):
    data = [{"text": "orphan text"}]
    result = cleaner.clean_list_data(data, case_id=1)

    assert result.cleaned_data == []
    assert result.cleaning_operations["removed_items"] == 1


# --- clean_string_data -----------------------------------------------------


def test_clean_string_parses_valid_json_array(cleaner):
    data = json.dumps(
        [
            {"bbox": [0, 0, 10, 10], "category": "Text", "text": "a"},
            {"bbox": [0, 0, 10, 10], "category": "Title", "text": "b"},
        ]
    )
    result = cleaner.clean_string_data(data, case_id=1)

    assert result.success is True
    assert len(result.cleaned_data) == 2


def test_clean_string_extracts_json_from_explanatory_text(cleaner):
    payload = '[{"bbox": [0, 0, 10, 10], "category": "Text", "text": "hi"}]'
    wrapped = f"Here is the result:\n{payload}\nLet me know if you need more."

    result = cleaner.clean_string_data(wrapped, case_id=1)

    assert result.success is True
    assert result.cleaning_operations["json_extracted"] is True
    assert len(result.cleaned_data) == 1
    assert result.cleaned_data[0]["category"] == "Text"


def test_clean_string_plain_text_returns_empty(cleaner):
    """Plain text (no JSON delimiters) is detected and short-circuits to []."""
    result = cleaner.clean_string_data("This is just OCR text with no structure.", case_id=1)

    assert result.success is True
    assert result.cleaned_data == []
    assert result.cleaning_operations["plain_text_detected"] is True


def test_clean_string_missing_delimiter_pattern_fires(cleaner):
    """The `}\\s*\\{(?!")` regex repairs `}{` only when the next char is NOT a quote.

    This is its narrow purpose — see the pattern definition; it's not meant
    for the common `}{"key":` case (which the JSON fallback path handles instead).
    """
    fixed, fixes = cleaner._fix_missing_delimiters("}{1: 2}")
    assert fixes == 1
    assert fixed == "},{1: 2}"

    # `}{"...` is intentionally left alone by this pattern.
    fixed2, fixes2 = cleaner._fix_missing_delimiters('}{"k": 1}')
    assert fixes2 == 0
    assert fixed2 == '}{"k": 1}'


def test_clean_string_recovers_dicts_when_outer_json_breaks(cleaner):
    """Even without a comma between objects, fallback regex extracts both dicts."""
    broken = (
        '[{"bbox": [0, 0, 1, 1], "category": "Text", "text": "a"}'
        '{"bbox": [0, 0, 1, 1], "category": "Text", "text": "b"}]'
    )
    result = cleaner.clean_string_data(broken, case_id=1)
    assert result.success is True
    assert len(result.cleaned_data) == 2
    assert [d["text"] for d in result.cleaned_data] == ["a", "b"]


def test_clean_string_removes_duplicate_complete_dicts(cleaner):
    dup = '{"bbox": [0, 0, 1, 1], "category": "Text", "text": "x"}'
    data = f"[{dup}, {dup}, {dup}]"
    result = cleaner.clean_string_data(data, case_id=1)

    assert result.success is True
    assert len(result.cleaned_data) == 1
    assert result.cleaning_operations["duplicate_dicts_removed"] == 2


def test_clean_string_fallback_extracts_dicts_from_broken_json(cleaner):
    """If the outer JSON is unparseable, fallback regex extraction kicks in."""
    # The trailing junk makes json.loads fail; cleaner should still recover the dicts.
    broken = (
        '[{"bbox": [0, 0, 1, 1], "category": "Text", "text": "a"},'
        '{"bbox": [2, 2, 3, 3], "category": "Text", "text": "b"}, !!!garbage!!!]'
    )
    result = cleaner.clean_string_data(broken, case_id=1)
    assert result.success is True
    assert result.cleaned_data == [
        {"bbox": [0, 0, 1, 1], "category": "Text", "text": "a"},
        {"bbox": [2, 2, 3, 3], "category": "Text", "text": "b"},
    ]


# --- _extract_outermost_json_array ----------------------------------------


def test_extract_outermost_json_array_returns_unchanged_when_invalid(cleaner):
    text, extracted = cleaner._extract_outermost_json_array("no brackets here")
    assert extracted is False
    assert text == "no brackets here"


def test_extract_outermost_json_array_picks_full_span(cleaner):
    text, extracted = cleaner._extract_outermost_json_array("prefix [1, 2, 3] suffix")
    assert extracted is True
    assert text == "[1, 2, 3]"


# --- _is_plain_text_output -------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("just plain text", True),
        ("text with { brace", False),
        ("text with [ bracket", False),
        ("", True),
    ],
)
def test_is_plain_text_output(cleaner, text, expected):
    assert cleaner._is_plain_text_output(text) is expected


# --- _ensure_json_format ---------------------------------------------------


def test_ensure_json_format_wraps_missing_brackets(cleaner):
    assert cleaner._ensure_json_format('{"a":1}') == '[{"a":1}]'


def test_ensure_json_format_strips_trailing_comma(cleaner):
    assert cleaner._ensure_json_format('[{"a":1},').endswith("]")
    assert "," not in cleaner._ensure_json_format('[{"a":1},')[-2:]


# --- remove_duplicate_category_text_pairs_and_bbox -----------------------


def test_remove_duplicate_bbox_pairs(cleaner):
    data = [
        {"bbox": [0, 0, 1, 1], "category": "Text", "text": "a"},
        {"bbox": [0, 0, 1, 1], "category": "Text", "text": "b"},
    ]
    out = cleaner.remove_duplicate_category_text_pairs_and_bbox(data, case_id=1)
    assert len(out) == 1
    assert out[0]["text"] == "a"


def test_remove_duplicate_category_text_pairs_threshold(cleaner):
    """Threshold is 5 occurrences for category-text pairs; 4 must NOT trigger."""
    base = {"bbox": [0, 0, 1, 1], "category": "Text", "text": "same"}
    # 4 distinct bboxes, same category+text → below threshold for cat-text but
    # we still need distinct bboxes so the bbox path doesn't fire either.
    data = [{**base, "bbox": [i, 0, 1, 1]} for i in range(4)]
    out = cleaner.remove_duplicate_category_text_pairs_and_bbox(data, case_id=1)
    assert len(out) == 4

    # At 5+ duplicates, all but the first are removed.
    data5 = [{**base, "bbox": [i, 0, 1, 1]} for i in range(5)]
    out5 = cleaner.remove_duplicate_category_text_pairs_and_bbox(data5, case_id=1)
    assert len(out5) == 1


def test_dedup_short_list_is_noop(cleaner):
    """Lists of size ≤1 skip dedup entirely."""
    single = [{"bbox": [0, 0, 1, 1], "category": "Text", "text": "a"}]
    assert cleaner.remove_duplicate_category_text_pairs_and_bbox(single, case_id=1) == single
    assert cleaner.remove_duplicate_category_text_pairs_and_bbox([], case_id=1) == []


# --- clean_model_output (high-level entrypoint) ---------------------------


def test_clean_model_output_with_list_input(cleaner):
    data = [{"bbox": [0, 0, 1, 1], "category": "Text", "text": "hi"}]
    out = cleaner.clean_model_output(data)
    assert isinstance(out, list)
    assert out[0]["text"] == "hi"


def test_clean_model_output_with_string_input(cleaner):
    out = cleaner.clean_model_output('[{"bbox": [0, 0, 1, 1], "category": "Text", "text": "hi"}]')
    assert isinstance(out, list)
    assert out[0]["category"] == "Text"


def test_clean_model_output_returns_input_on_total_failure(cleaner, monkeypatch):
    """If the inner cleaner raises, return the raw input rather than crash."""

    def boom(*a, **kw):
        raise RuntimeError("boom")

    monkeypatch.setattr(cleaner, "clean_list_data", boom)
    out = cleaner.clean_model_output([1, 2, 3])
    assert out == [1, 2, 3]


# --- CleanedData dataclass smoke ------------------------------------------


def test_cleaned_data_fields_round_trip():
    cd = CleanedData(
        case_id=7,
        original_type="list",
        original_length=3,
        cleaned_data=[{"a": 1}],
        cleaning_operations={"type": "list"},
        success=True,
    )
    assert cd.case_id == 7
    assert cd.success is True
    assert cd.cleaned_data == [{"a": 1}]
