# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
Schema Chunker - Standalone utility for splitting large JSON schemas.

Main functions:
1. split_schema() - Splits large schemas into manageable chunks with optimal field packing
"""

from typing import Dict, Any, List


def count_fields(schema: Dict[str, Any]) -> int:
    """Count total number of fields in a schema recursively."""
    if not isinstance(schema, dict):
        return 0

    count = 0
    if "properties" in schema:
        for prop_schema in schema["properties"].values():
            count += 1
            count += count_fields(prop_schema)

    if "items" in schema:
        count += count_fields(schema["items"])

    return count


def split_schema(schema: Dict[str, Any], max_fields: int = 100) -> List[Dict[str, Any]]:
    """
    Split a large schema into smaller chunks.

    Args:
        schema: The JSON schema to split
        max_fields: Maximum number of fields per chunk

    Returns:
        List of schema chunks
    """
    if not schema.get("properties"):
        return [schema]

    # Analyze field distribution
    property_fields = {}
    for prop_name, prop_schema in schema["properties"].items():
        property_fields[prop_name] = count_fields(prop_schema) + 1

    total_fields = sum(property_fields.values())

    if total_fields <= max_fields:
        return [schema]

    # Split into chunks
    chunks = []
    current_chunk = {
        "type": schema.get("type", "object"),
        "properties": {},
        "required": [],
        "additionalProperties": schema.get("additionalProperties", False),
    }

    # Copy metadata
    for key in ["$schema", "description", "title"]:
        if key in schema:
            current_chunk[key] = schema[key]

    current_field_count = 0
    required_fields = set(schema.get("required", []))

    for prop_name, prop_schema in schema["properties"].items():
        prop_field_count = property_fields[prop_name]

        # Start new chunk if adding this property would exceed limit
        if current_field_count + prop_field_count > max_fields and current_chunk["properties"]:
            chunks.append(current_chunk)
            current_chunk = {
                "type": schema.get("type", "object"),
                "properties": {},
                "required": [],
                "additionalProperties": schema.get("additionalProperties", False),
            }
            # Copy metadata
            for key in ["$schema", "description", "title"]:
                if key in schema:
                    current_chunk[key] = schema[key]
            current_field_count = 0

        # Add property to current chunk
        current_chunk["properties"][prop_name] = prop_schema
        if prop_name in required_fields:
            current_chunk["required"].append(prop_name)
        current_field_count += prop_field_count

    # Add final chunk
    if current_chunk["properties"]:
        chunks.append(current_chunk)

    return chunks
