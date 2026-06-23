# SPDX-License-Identifier: Apache-2.0
"""Convert Pydantic models to JSON schemas."""

from __future__ import annotations
from typing import Any, Literal
import re
from tensorlake_docai.extraction.schema_enricher_utils import inline_refs

Flavors = Literal["vanilla", "openai_output_schema", "openai_tool_schema"]


def enrich_for_openai(
    schema: dict[str, Any], is_top_level: bool = True, enforce_required: bool = True
) -> None:
    """Recursively enrich schema with OpenAI-compatible behaviors."""
    schema_type = schema.get("type")

    # Clean up problematic fields that cause OpenAI validation errors
    if "enum" in schema and (
        schema["enum"] == "" or schema["enum"] is None or schema["enum"] == []
    ):
        del schema["enum"]
    if "description" in schema and (schema["description"] == "" or schema["description"] is None):
        del schema["description"]
    if "default" in schema and schema["default"] is None:
        del schema["default"]

    if schema_type == "object" or isinstance(schema_type, list) and "object" in schema_type:
        schema["additionalProperties"] = False

        # Ensure ALL objects have a 'required' field (OpenAI requirement)
        if "properties" in schema and isinstance(schema["properties"], dict):
            valid_properties = set(schema["properties"].keys())
            existing_required = set(schema.get("required", []))
            # Only keep required fields that actually have property definitions
            valid_required = existing_required.intersection(valid_properties)

            if enforce_required:
                # Combine valid existing required fields with all properties, making all fields required by default
                schema["required"] = sorted(list(valid_required.union(valid_properties)))
            else:
                # Just keep the valid required fields
                schema["required"] = sorted(list(valid_required))
        else:
            # No properties defined, so ensure empty required array exists
            schema["required"] = []

        # Recursively process nested properties
        if "properties" in schema and isinstance(schema["properties"], dict):
            for prop in schema["properties"].values():
                if isinstance(prop, dict):
                    enrich_for_openai(prop, is_top_level=False, enforce_required=enforce_required)

        if not is_top_level:
            schema["type"] = ["object", "null"]

    elif schema_type == "array":
        schema["type"] = ["array", "null"]
        if "items" in schema and isinstance(schema["items"], dict):
            enrich_for_openai(
                schema["items"], is_top_level=False, enforce_required=enforce_required
            )

    elif isinstance(schema_type, str) and schema_type not in ["object", "array", "null"]:
        schema["type"] = [schema_type, "null"]

    elif isinstance(schema_type, list) and "null" not in schema_type:
        schema_type.append("null")

    # Recurse into definitions
    for key in ["$defs", "definitions"]:
        if key in schema and isinstance(schema[key], dict):
            for sub in schema[key].values():
                if isinstance(sub, dict):
                    enrich_for_openai(sub, is_top_level=False, enforce_required=enforce_required)

    # Handle combiners
    for combiner in ["allOf", "anyOf", "oneOf"]:
        if combiner in schema and isinstance(schema[combiner], list):
            for subschema in schema[combiner]:
                if isinstance(subschema, dict):
                    enrich_for_openai(
                        subschema, is_top_level=False, enforce_required=enforce_required
                    )


def _sanitize_schema_name(raw_name: str) -> str:
    """Sanitize schema name to match ^[a-zA-Z0-9_-]+$ required by OpenAI."""
    if not isinstance(raw_name, str):
        return "output"
    name = re.sub(r"[^a-zA-Z0-9_-]+", "_", raw_name)
    name = re.sub(r"_+", "_", name).strip("_")
    return name or "output"


def pydantic_converter(
    model: dict[str, Any],
    include_title: bool = False,
    flavor: Flavors = "vanilla",
    enrich_openai: bool = True,
    enforce_required: bool = True,
) -> dict[str, Any]:
    """Convert a Pydantic model to an inline JSON schema.

    Args:
        model: The Pydantic model to convert.
        include_title: Whether to include the `title` fields in the schema.
        flavor: The flavor of the output schema.

    Returns:
        The inline JSON schema.
    """
    schema = inline_refs(model, include_title=include_title)

    if enrich_openai:
        enrich_for_openai(schema, is_top_level=True, enforce_required=enforce_required)

    match flavor:
        case "openai_output_schema":
            schema["name"] = _sanitize_schema_name(schema.pop("title", "output"))
            schema["strict"] = True
            # Preserve all top-level keywords by moving the entire schema under "schema",
            # but exclude meta keys like $schema from being moved inside.
            inner = {
                k: v
                for k, v in schema.items()
                if k not in ("name", "strict", "schema") and k != "$schema"
            }
            # Ensure sensible defaults for inner schema when missing
            if "type" not in inner:
                inner["type"] = "object"
                inner.setdefault("properties", {})
                inner.setdefault("additionalProperties", False)
                inner.setdefault("required", [])
            schema["schema"] = inner
            # Remove other top-level keys to avoid duplication
            for k in list(schema.keys()):
                if k not in ("name", "strict", "schema"):
                    del schema[k]
        case "openai_tool_schema":
            schema["name"] = _sanitize_schema_name(schema.pop("title", "tool"))
            schema["strict"] = True
            # Preserve all top-level keywords by moving the entire schema under "parameters",
            # but exclude meta keys like $schema from being moved inside.
            inner = {
                k: v
                for k, v in schema.items()
                if k not in ("name", "strict", "parameters") and k != "$schema"
            }
            # Ensure sensible defaults for inner schema when missing
            if "type" not in inner:
                inner["type"] = "object"
                inner.setdefault("properties", {})
                inner.setdefault("additionalProperties", False)
                inner.setdefault("required", [])
            schema["parameters"] = inner
            # Remove other top-level keys to avoid duplication
            for k in list(schema.keys()):
                if k not in ("name", "strict", "parameters"):
                    del schema[k]
        case _:
            pass
    schema.pop("$schema", None)
    return schema
