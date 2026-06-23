# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
Shared error utilities.
"""

import re
from typing import Any


# This is for azure OCR error message extraction
def extract_provider_error_message(err: BaseException) -> str:
    """Extract and sanitize a user-friendly error message from provider exceptions.

    Surfaces messages from nested structures like error.innererror.message and removes
    provider-identifying words/domains from the surfaced message.
    """

    def pick_error(obj: Any, name: str):
        if obj is None:
            return None
        if isinstance(obj, dict):
            return obj.get(name)
        return getattr(obj, name, None)

    try:
        err_obj = pick_error(err, "error")
        inner = pick_error(err_obj, "innererror")
        message = (
            pick_error(inner, "message")
            or pick_error(err_obj, "message")
            or getattr(err, "message", None)
            or str(err)
        )
    except Exception:
        message = str(err)

    try:
        # Remove provider-specific names/domains from surfaced message
        _s = re.sub(
            r"cognitiveservices\\.azure\\.com", "ocr01.com", str(message or ""), flags=re.IGNORECASE
        )
        _msg = re.sub(
            r"\\b(azure|microsoft|document\\s*intelligence)\\b", "", _s, flags=re.IGNORECASE
        )
        message = re.sub(r"\\s+", " ", _msg).strip()
    except Exception:
        message = str(message)

    return message
