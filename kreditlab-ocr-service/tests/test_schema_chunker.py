# SPDX-License-Identifier: Apache-2.0
"""Tests for schema_chunker — splits large JSON schemas into bounded chunks."""

from tensorlake_docai.extraction.schema_chunker import count_fields, split_schema

# --- count_fields ---------------------------------------------------------


def test_count_fields_empty_schema():
    assert count_fields({}) == 0


def test_count_fields_non_dict_input():
    assert count_fields(None) == 0
    assert count_fields("not a schema") == 0
    assert count_fields(42) == 0


def test_count_fields_flat_object():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}, "b": {"type": "integer"}},
    }
    assert count_fields(schema) == 2


def test_count_fields_nested_object():
    schema = {
        "type": "object",
        "properties": {
            "outer": {
                "type": "object",
                "properties": {"x": {"type": "string"}, "y": {"type": "string"}},
            }
        },
    }
    # 1 (outer) + 2 (x, y) = 3
    assert count_fields(schema) == 3


def test_count_fields_array_of_objects():
    schema = {
        "type": "object",
        "properties": {
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {"a": {"type": "string"}, "b": {"type": "string"}},
                },
            }
        },
    }
    # 1 (items) + 2 (a, b) = 3
    assert count_fields(schema) == 3


# --- split_schema ---------------------------------------------------------


def test_split_schema_no_properties_returns_unchanged():
    schema = {"type": "string"}
    assert split_schema(schema) == [schema]


def test_split_schema_small_schema_returns_single_chunk():
    schema = {
        "type": "object",
        "properties": {"a": {"type": "string"}},
    }
    chunks = split_schema(schema, max_fields=10)
    assert len(chunks) == 1
    assert chunks[0]["properties"]["a"] == {"type": "string"}


def test_split_schema_oversize_splits_into_multiple_chunks():
    props = {f"f{i}": {"type": "string"} for i in range(10)}
    schema = {"type": "object", "properties": props}
    chunks = split_schema(schema, max_fields=3)

    assert len(chunks) >= 4
    # Property names are partitioned, no field appears in two chunks
    seen = set()
    for chunk in chunks:
        for name in chunk["properties"]:
            assert name not in seen, f"{name} duplicated across chunks"
            seen.add(name)
    assert seen == set(props.keys())


def test_split_schema_preserves_metadata_per_chunk():
    schema = {
        "type": "object",
        "title": "BigSchema",
        "description": "Lots of fields",
        "properties": {f"f{i}": {"type": "string"} for i in range(6)},
    }
    chunks = split_schema(schema, max_fields=2)

    assert len(chunks) >= 3
    for chunk in chunks:
        assert chunk["title"] == "BigSchema"
        assert chunk["description"] == "Lots of fields"
        assert chunk["additionalProperties"] is False


def test_split_schema_routes_required_fields_to_correct_chunks():
    schema = {
        "type": "object",
        "properties": {f"f{i}": {"type": "string"} for i in range(6)},
        "required": ["f0", "f3"],
    }
    chunks = split_schema(schema, max_fields=2)

    # f0 ends up in a chunk whose required list contains it
    for chunk in chunks:
        if "f0" in chunk["properties"]:
            assert "f0" in chunk["required"]
        if "f3" in chunk["properties"]:
            assert "f3" in chunk["required"]
        # No required entries point at fields outside this chunk
        for req in chunk["required"]:
            assert req in chunk["properties"]
