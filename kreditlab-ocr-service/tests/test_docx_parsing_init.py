# SPDX-License-Identifier: Apache-2.0
"""Smoke tests for the docx_parsing package __init__ module."""


def test_import_process_docx_to_structured_pages():
    from tensorlake_docai.pipeline.docx_parsing import process_docx_to_structured_pages

    assert callable(process_docx_to_structured_pages)


def test_all_exports():
    import tensorlake_docai.pipeline.docx_parsing as pkg

    assert "process_docx_to_structured_pages" in pkg.__all__
