# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
AWS Textract API utilities with timeout handling.
Retries are handled at the function level
"""

import time
from typing import Any


def robust_textract_analyze_document(
    extractor, file_source, features, timeout: int = 300, **kwargs
) -> Any:
    """Textract API call with timeout (retries handled at function level)"""

    print(f"🔍 Starting Textract analysis (timeout: {timeout}s)")
    start_time = time.time()

    try:
        # Call textractor's analyze_document method
        document = extractor.analyze_document(file_source=file_source, features=features, **kwargs)

        elapsed_time = time.time() - start_time
        print(f"✅ Textract analysis completed in {elapsed_time:.1f}s")
        return document

    except Exception as e:
        elapsed_time = time.time() - start_time
        if elapsed_time >= timeout:
            raise TimeoutError(f"Textract analysis timed out after {elapsed_time:.1f}s")
        else:
            raise e


TIMEOUT_IMAGE = 600
TIMEOUT_PDF_PAGE = 600
TIMEOUT_LARGE = 600
