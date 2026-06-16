"""
pdf_password_resolver.py

Resolves the password for a PDF file at a given path, for offline scripts
(audit, validation, regression tests) that can't prompt the user interactively.

Lookup chain (first match wins):
  1. Environment variable:  PDF_PASSWORD__<sanitized_stem>
  2. Pattern match in:      .pdf_passwords.json  (gitignored, never committed)
  3. Empty string           (lets pdf_security handle "encrypted but no-password-needed")

Format of .pdf_passwords.json:
  {
    "Bank-Statement/BankIslam/Mytutor/*.pdf": "customer_password_here",
    "Bank-Statement/SomeBank/OtherCustomer/**.pdf": "other_password"
  }

Keys are fnmatch-style glob patterns (relative to repo root).
"""

from __future__ import annotations

import fnmatch
import json
import os
import re
from pathlib import Path
from typing import Optional

_CONFIG_FILENAME = ".pdf_passwords.json"


def _repo_root() -> Path:
    return Path(__file__).resolve().parent


def _env_key(pdf_path: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9]+", "_", pdf_path.stem).strip("_").upper()
    return f"PDF_PASSWORD__{stem}"


def _load_config() -> dict:
    cfg_path = _repo_root() / _CONFIG_FILENAME
    if not cfg_path.exists():
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
    except Exception:
        pass
    return {}


def resolve_password(pdf_path: Path) -> Optional[str]:
    """
    Return the password for this PDF, or None if no configured password found.
    Callers should still try empty-password first (pdf_security handles that).
    """
    pdf_path = Path(pdf_path)

    # 1. Environment variable lookup (per-file)
    env_key = _env_key(pdf_path)
    if env_key in os.environ:
        return os.environ[env_key]

    # 2. Glob patterns in config file
    config = _load_config()
    if not config:
        return None

    root = _repo_root()
    try:
        rel = pdf_path.resolve().relative_to(root)
        rel_str = str(rel).replace(os.sep, "/")
    except Exception:
        rel_str = str(pdf_path).replace(os.sep, "/")

    for pattern, password in config.items():
        if not isinstance(password, str):
            continue
        normalized_pattern = pattern.replace(os.sep, "/")
        if fnmatch.fnmatch(rel_str, normalized_pattern):
            return password
        # Also try matching just the filename for convenience
        if fnmatch.fnmatch(pdf_path.name, normalized_pattern):
            return password

    return None


def read_pdf_bytes_decrypted(pdf_path: Path) -> bytes:
    """
    Read a PDF from disk and return decrypted bytes ready for any parser.
    Uses pdf_security.decrypt_pdf_bytes under the hood.

    Raises ValueError with a clean message if password is required but none
    is configured, or if the configured password is wrong.
    """
    from pdf_security import is_pdf_encrypted, decrypt_pdf_bytes

    pdf_bytes = Path(pdf_path).read_bytes()

    if not is_pdf_encrypted(pdf_bytes):
        return pdf_bytes

    password = resolve_password(Path(pdf_path))
    if password is None:
        raise ValueError(
            f"PDF is password-protected and no password is configured for: {pdf_path}. "
            f"Add an entry to .pdf_passwords.json or set {_env_key(Path(pdf_path))} env var."
        )

    return decrypt_pdf_bytes(pdf_bytes, password)
