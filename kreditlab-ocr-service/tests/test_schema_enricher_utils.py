# SPDX-License-Identifier: Apache-2.0
"""Schema enrichment helpers — these run on every structured-extraction
request, so the small primitives must stay correct."""

from tensorlake_docai.extraction.schema_enricher_utils import (
    inline_refs,
    is_array_type,
    is_object_type,
    is_simple_schema,
)


def test_is_object_type():
    assert is_object_type("object")
    assert is_object_type(["object", "null"])
    assert not is_object_type("string")
    assert not is_object_type(["string", "number"])


def test_is_array_type():
    assert is_array_type("array")
    assert is_array_type(["array", "null"])
    assert not is_array_type("object")


def test_is_simple_schema_primitives():
    assert is_simple_schema({"type": "string"})
    assert is_simple_schema({"type": "integer"})
    assert is_simple_schema({"type": ["string", "null"]})


def test_is_simple_schema_enum_and_const():
    assert is_simple_schema({"enum": ["a", "b"]})
    assert is_simple_schema({"const": 42})


def test_is_simple_schema_anyof_with_primitive():
    assert is_simple_schema({"anyOf": [{"type": "string"}, {"type": "null"}]})
    assert is_simple_schema({"oneOf": [{"type": "integer"}]})


def test_is_simple_schema_object_is_not_simple():
    assert not is_simple_schema({"type": "object", "properties": {}})
    assert not is_simple_schema({"type": "array", "items": {"type": "string"}})


def test_is_simple_schema_non_dict():
    # Defensive branch — the helper accepts arbitrary input and returns False.
    assert not is_simple_schema("string")  # type: ignore[arg-type]
    assert not is_simple_schema(None)  # type: ignore[arg-type]


def test_inline_refs_basic():
    schema = {
        "type": "object",
        "properties": {"address": {"$ref": "#/$defs/Address"}},
        "$defs": {
            "Address": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
            }
        },
    }
    out = inline_refs(schema)
    assert "$defs" not in out
    assert out["properties"]["address"]["type"] == "object"
    assert out["properties"]["address"]["properties"]["city"]["type"] == "string"


def test_inline_refs_supports_definitions_alias():
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/definitions/X"}},
        "definitions": {"X": {"type": "integer"}},
    }
    out = inline_refs(schema)
    assert "definitions" not in out
    assert out["properties"]["x"] == {"type": "integer"}


def test_inline_refs_preserves_siblings_on_ref():
    schema = {
        "type": "object",
        "properties": {
            "x": {"$ref": "#/$defs/X", "description": "kept"},
        },
        "$defs": {"X": {"type": "string"}},
    }
    out = inline_refs(schema)
    assert out["properties"]["x"]["type"] == "string"
    assert out["properties"]["x"]["description"] == "kept"


def test_inline_refs_strips_title_by_default():
    schema = {
        "title": "Outer",
        "type": "object",
        "properties": {"x": {"type": "string", "title": "InnerX"}},
    }
    out = inline_refs(schema)
    assert "title" not in out
    assert "title" not in out["properties"]["x"]


def test_inline_refs_keeps_title_when_requested():
    schema = {
        "title": "Outer",
        "type": "object",
        "properties": {"x": {"type": "string", "title": "InnerX"}},
    }
    out = inline_refs(schema, include_title=True)
    assert out["title"] == "Outer"
    assert out["properties"]["x"]["title"] == "InnerX"


def test_inline_refs_does_not_mutate_input():
    schema = {
        "type": "object",
        "properties": {"x": {"$ref": "#/$defs/X"}},
        "$defs": {"X": {"type": "integer"}},
    }
    snapshot = repr(schema)
    inline_refs(schema)
    assert repr(schema) == snapshot
