"""
pdf_security.py

Utilities for handling password-protected (encrypted) PDFs.

Important nuance:
- Some PDFs have an /Encrypt dictionary (reader.is_encrypted == True) but can be opened
  with an empty password (""). In UX terms these should NOT be treated as password-protected.
- Therefore, `is_pdf_encrypted()` below means: "does this PDF REQUIRE a password from the user?"

We decrypt into a new in-memory PDF (bytes) so downstream libraries like pdfplumber
can read it without needing to pass passwords around.
"""

from __future__ import annotations

from io import BytesIO
from typing import Optional

from pypdf import PdfReader, PdfWriter


def _can_open_with_password(reader: PdfReader, password: str) -> bool:
    """
    Return True if pypdf can decrypt and access pages using the provided password.
    """
    try:
        result = reader.decrypt(password)
    except Exception:
        return False

    if result == 0:
        return False

    # Some PDFs report successful decrypt but still fail on page access if wrong.
    try:
        _ = len(reader.pages)
        if len(reader.pages) > 0:
            _ = reader.pages[0]  # touch first page
    except Exception:
        return False

    return True


def is_pdf_encrypted(pdf_bytes: bytes) -> bool:
    """
    Return True only if the PDF requires a user password to be opened.

    Notes:
    - If the PDF is encrypted but opens with an empty password, this returns False
      (so the UI won't incorrectly ask the user for a password).
    - If the PDF cannot be parsed at all, we return False to avoid false positives
      (the downstream parser will raise a readable error anyway).
    """
    try:
        reader = PdfReader(BytesIO(pdf_bytes), strict=False)
    except Exception:
        # Not parseable (or not a PDF). Do NOT treat as encrypted.
        return False

    if not bool(getattr(reader, "is_encrypted", False)):
        return False

    # If it can be opened with empty password, it is not "password protected" for users.
    if _can_open_with_password(reader, ""):
        return False

    # Otherwise it is encrypted and requires a password.
    return True


def decrypt_pdf_bytes(pdf_bytes: bytes, password: Optional[str]) -> bytes:
    """
    Decrypt an encrypted PDF and return a decrypted PDF as bytes.

    Behavior:
    - If PDF is not encrypted -> returns original bytes.
    - If PDF is encrypted but opens with empty password -> returns decrypted bytes without requiring user input.
    - If PDF requires a password:
        - raises ValueError if password is missing
        - raises ValueError if password is incorrect
    """
    reader = PdfReader(BytesIO(pdf_bytes), strict=False)
    if not reader.is_encrypted:
        return pdf_bytes

    # Try empty password first (covers "encrypted but no user password needed")
    if _can_open_with_password(reader, ""):
        writer = PdfWriter()
        for page in reader.pages:
            writer.add_page(page)
        out = BytesIO()
        writer.write(out)
        return out.getvalue()

    # Now it truly requires a password
    if not password:
        raise ValueError("Password required for encrypted PDF.")

    # Re-read for a clean decrypt attempt (since PdfReader.decrypt mutates state)
    reader = PdfReader(BytesIO(pdf_bytes), strict=False)
    if not _can_open_with_password(reader, password):
        raise ValueError("Incorrect password or unable to decrypt PDF.")

    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)

    out = BytesIO()
    writer.write(out)
    return out.getvalue()
