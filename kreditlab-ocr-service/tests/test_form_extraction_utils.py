# SPDX-License-Identifier: Apache-2.0
"""Form JSON → Markdown conversion. Pure function; runs in the form-extraction
post-processing path."""

import json

from tensorlake_docai.extraction.form_extraction_utils import convert_form_json_to_markdown


def test_dict_input_renders_key_value_pairs():
    out = convert_form_json_to_markdown(json.dumps({"name": "Alice", "age": 30}))
    assert "**name**: Alice" in out
    assert "**age**: 30" in out


def test_list_of_field_value_items():
    items = [
        {"box_id": "b1", "field_name": "Name", "type": "text", "value": "Alice"},
        {"field_name": "Subscribed", "type": "checkbox", "value": "yes"},
    ]
    out = convert_form_json_to_markdown(json.dumps(items))
    assert "[b1] **Name** (text): Alice" in out
    assert "**Subscribed** (checkbox): yes" in out


def test_generic_key_value_fallback():
    items = [{"key": "Phone", "value": "555-0100"}, {"label": "Email", "text": "a@b.com"}]
    out = convert_form_json_to_markdown(json.dumps(items))
    assert "**Phone**: 555-0100" in out
    assert "**Email**: a@b.com" in out


def test_invalid_json_passed_through():
    raw = "not valid json {{{"
    assert convert_form_json_to_markdown(raw) == raw


def test_accepts_already_parsed_dict():
    out = convert_form_json_to_markdown({"x": "y"})  # type: ignore[arg-type]
    assert "**x**: y" in out
