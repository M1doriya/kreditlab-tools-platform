# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations
import copy
from typing import Any

"""Shared helpers for JSON Schema enrichment and inlining."""

PRIMITIVE_TYPES = {"string", "number", "integer", "boolean"}


def is_object_type(t: Any) -> bool:
    return t == "object" or (isinstance(t, list) and "object" in t)


def is_array_type(t: Any) -> bool:
    return t == "array" or (isinstance(t, list) and "array" in t)


def is_simple_schema(node: dict[str, Any]) -> bool:
    """Return True for primitive fields (incl. Optional via anyOf/oneOf)."""
    if not isinstance(node, dict):
        return False
    t = node.get("type")
    if isinstance(t, list) and any(tt in PRIMITIVE_TYPES for tt in t):
        return True
    if t in PRIMITIVE_TYPES:
        return True
    for key in ("anyOf", "oneOf"):
        variants = node.get(key)
        if isinstance(variants, list):
            for opt in variants:
                if isinstance(opt, dict) and opt.get("type") in PRIMITIVE_TYPES:
                    return True
    if ("enum" in node) or ("const" in node):
        return True
    return False


def _extract_definitions(original: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return (schema_wo_defs, defs) without mutating caller's input."""
    base = copy.deepcopy(original)
    defs: dict[str, Any] = {}
    if isinstance(base.get("definitions"), dict):
        defs.update(copy.deepcopy(base["definitions"]))
        base.pop("definitions", None)
    if isinstance(base.get("$defs"), dict):
        defs.update(copy.deepcopy(base["$defs"]))
        base.pop("$defs", None)
    return base, defs


def _ref_key_from_path(ref_path: str) -> str | None:
    if ref_path.startswith("#/$defs/"):
        return ref_path.split("#/$defs/")[-1]
    if ref_path.startswith("#/definitions/"):
        return ref_path.split("#/definitions/")[-1]
    return None


def _inline_node(obj: Any, defs: dict[str, Any], include_title: bool) -> Any:
    if isinstance(obj, dict):
        if "$ref" in obj:
            key = _ref_key_from_path(obj["$ref"]) or ""
            resolved = copy.deepcopy(defs.get(key, {}))
            siblings = {k: v for k, v in obj.items() if k != "$ref"}
            return _inline_node({**resolved, **siblings}, defs, include_title)
        out: dict[str, Any] = {}
        for k, v in obj.items():
            if k == "title":
                if include_title:
                    out[k] = v
            else:
                out[k] = _inline_node(v, defs, include_title)
        return out
    if isinstance(obj, list):
        return [_inline_node(x, defs, include_title) for x in obj]
    return obj


def inline_refs(schema: dict[str, Any], include_title: bool = False) -> dict[str, Any]:
    """Inline $ref from $defs/definitions and preserve siblings (no mutation)."""
    if not isinstance(schema, dict):
        return schema
    base, defs = _extract_definitions(schema)
    title = base.get("title")
    inlined = _inline_node(base, defs, include_title)
    if include_title and title:
        inlined["title"] = title
    else:
        inlined.pop("title", None)
    return inlined
