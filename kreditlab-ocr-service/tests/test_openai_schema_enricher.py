# SPDX-License-Identifier: Apache-2.0
"""Tests for openai_schema_enricher — Pydantic/JSON-schema munging for the
strict OpenAI structured-output format.
"""

import pytest

from tensorlake_docai.extraction.openai_schema_enricher import (
    _sanitize_schema_name,
    enrich_for_openai,
    pydantic_converter,
)

# --- _sanitize_schema_name -----------------------------------------------


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("My Schema", "My_Schema"),
        ("clean-name_42", "clean-name_42"),
        ("foo!!!bar", "foo_bar"),
        ("  weird///name  ", "weird_name"),
        ("", "output"),
        ("!!!", "output"),
    ],
)
def test_sanitize_schema_name(raw, expected):
    assert _sanitize_schema_name(raw) == expected


def test_sanitize_schema_name_non_string():
    assert _sanitize_schema_name(None) == "output"
    assert _sanitize_schema_name(42) == "output"


# --- enrich_for_openai ----------------------------------------------------


def test_enrich_drops_empty_enum_description_and_null_default():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string", "enum": [], "description": "", "default": None}},
    }
    enrich_for_openai(schema)
    a = schema["properties"]["a"]
    assert "enum" not in a
    assert "description" not in a
    assert "default" not in a


def test_enrich_sets_additional_properties_false_and_required_all():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
    }
    enrich_for_openai(schema)
    assert schema["additionalProperties"] is False
    assert set(schema["required"]) == {"a", "b"}


def test_enrich_enforce_required_false_keeps_only_existing_required():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
        "required": ["a"],
    }
    enrich_for_openai(schema, enforce_required=False)
    assert schema["required"] == ["a"]


def test_enrich_filters_required_to_only_valid_property_names():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
        "required": ["a", "ghost"],
    }
    enrich_for_openai(schema, enforce_required=False)
    assert schema["required"] == ["a"]


def test_enrich_promotes_scalar_to_nullable_at_inner_level():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }
    enrich_for_openai(schema)
    # Inner string becomes nullable union.
    a_type = schema["properties"]["a"]["type"]
    assert a_type == ["string", "null"]


def test_enrich_top_level_object_type_not_nullable():
    schema = {"type": "object", "properties": {"a": {"type": "string"}}}
    enrich_for_openai(schema, is_top_level=True)
    # Top-level object type stays a plain string, not a list-with-null.
    assert schema["type"] == "object"


def test_enrich_makes_array_type_nullable_and_recurses_into_items():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"x": {"type": "string"}},
                },
            }
        },
    }
    enrich_for_openai(schema)
    arr = schema["properties"]["items"]
    assert arr["type"] == ["array", "null"]
    inner = arr["items"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["x"]
    assert inner["properties"]["x"]["type"] == ["string", "null"]


def test_enrich_recurses_through_defs():
    schema = {
        "type": "object",
        "properties": {},
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
            }
        },
    }
    enrich_for_openai(schema)
    inner = schema["$defs"]["Inner"]
    assert inner["additionalProperties"] is False
    assert inner["required"] == ["a"]
    assert inner["properties"]["a"]["type"] == ["string", "null"]


def test_enrich_recurses_into_anyOf_combiners():
    schema = {
        "type": "object",
        "properties": {
            "u": {
                "anyOf": [
                    {"type": "object", "properties": {"x": {"type": "string"}}},
                    {"type": "string"},
                ]
            }
        },
    }
    enrich_for_openai(schema)
    obj_variant = schema["properties"]["u"]["anyOf"][0]
    assert obj_variant["additionalProperties"] is False
    assert obj_variant["required"] == ["x"]


def test_enrich_handles_schema_with_no_properties():
    schema = {"type": "object"}
    enrich_for_openai(schema)
    assert schema["required"] == []
    assert schema["additionalProperties"] is False


# --- pydantic_converter end-to-end ---------------------------------------


def test_pydantic_converter_vanilla_passthrough():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }
    out = pydantic_converter(schema, flavor="vanilla")
    assert out["additionalProperties"] is False
    assert set(out["required"]) == {"a"}
    assert "$schema" not in out


def test_pydantic_converter_openai_output_schema_shape():
    schema = {
        "title": "My Output",
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }
    out = pydantic_converter(schema, flavor="openai_output_schema", include_title=True)
    assert out["name"] == "My_Output"
    assert out["strict"] is True
    assert "schema" in out
    assert out["schema"]["type"] == "object"
    assert out["schema"]["additionalProperties"] is False
    assert set(out["schema"]["required"]) == {"a"}


def test_pydantic_converter_openai_tool_schema_shape():
    schema = {
        "title": "Tool",
        "type": "object",
        "properties": {"x": {"type": "integer"}},
    }
    out = pydantic_converter(schema, flavor="openai_tool_schema", include_title=True)
    assert out["name"] == "Tool"
    assert out["strict"] is True
    assert "parameters" in out
    assert out["parameters"]["type"] == "object"


def test_pydantic_converter_inlines_refs():
    schema = {
        "type": "object",
        "properties": {"inner": {"$ref": "#/$defs/Inner"}},
        "$defs": {
            "Inner": {
                "type": "object",
                "properties": {"a": {"type": "string"}},
            }
        },
    }
    out = pydantic_converter(schema, flavor="vanilla")
    # $defs is gone after inlining.
    assert "$defs" not in out
    # `inner` is inlined to the object schema (and rewritten to nullable union).
    inner = out["properties"]["inner"]
    assert inner.get("properties", {}).get("a", {}).get("type") == ["string", "null"]
