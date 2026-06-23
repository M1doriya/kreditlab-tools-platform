# SPDX-License-Identifier: Apache-2.0
"""Tests for prompt constants in prompts/form_prompt.py."""

from tensorlake_docai.prompts.form_prompt import (
    FIGURE_FORM_GENERIC_PROMPT,
    FIGURE_FORM_TEMPLATE_PROMPT,
    FORM_TEMPLATES,
    FORM_TYPE_DETECTION_PROMPT,
)


def test_generic_prompt_is_non_empty_string():
    assert isinstance(FIGURE_FORM_GENERIC_PROMPT, str)
    assert len(FIGURE_FORM_GENERIC_PROMPT) > 0


def test_template_prompt_contains_placeholder():
    assert "{field_list}" in FIGURE_FORM_TEMPLATE_PROMPT


def test_detection_prompt_lists_known_form_types():
    for form_type in ("1040", "W2", "CMS_1500", "I9", "DS_11"):
        assert form_type in FORM_TYPE_DETECTION_PROMPT


def test_form_templates_contains_expected_keys():
    expected = {
        "1040",
        "W2",
        "1099_NEC",
        "1099_MISC",
        "CMS_1500",
        "UB_04",
        "I9",
        "W4",
        "DS_11",
        "DS_82",
    }
    assert expected.issubset(set(FORM_TEMPLATES.keys()))


def test_form_templates_values_are_nonempty_lists():
    for key, fields in FORM_TEMPLATES.items():
        assert isinstance(fields, list), f"{key} should map to a list"
        assert len(fields) > 0, f"{key} field list should not be empty"


def test_form_templates_values_are_strings():
    for key, fields in FORM_TEMPLATES.items():
        for field in fields:
            assert isinstance(field, str), f"Field {field!r} in {key} should be a string"
