# SPDX-License-Identifier: Apache-2.0
#!/usr/bin/env python3
"""
Azure API utilities with timeout handling and transient-aware retries.
"""

import random
import time
from typing import Any

from azure.core.exceptions import (  # type: ignore
    HttpResponseError,
    ServiceRequestError,
    ServiceResponseError,
)
from requests.exceptions import ReadTimeout, SSLError, ConnectionError  # type: ignore
from tensorlake.applications import RequestError as RequestException  # type: ignore
from tensorlake_docai.providers.error_utils import extract_provider_error_message


def _is_http_transient(err: BaseException) -> bool:
    """Classify errors that are likely transient and worth retrying."""
    if isinstance(err, (ServiceRequestError, ServiceResponseError)):
        return True
    if isinstance(err, (ReadTimeout, SSLError, ConnectionError, TimeoutError)):
        return True
    if isinstance(err, HttpResponseError):
        # Retry on 408, 429, and 5xx
        status = None
        try:
            status = err.status_code
        except Exception:
            status = None
        return bool(status in (408, 429) or (isinstance(status, int) and 500 <= status <= 599))
    return False


def robust_azure_analyze_document(
    client, model_id: str, request, timeout: int = 300, **kwargs
) -> Any:
    """Azure Document Intelligence API call with timeout and retries."""

    max_attempts = 10
    base_delay = 1
    max_delay = 10

    print(f"[ARU] Starting Azure analysis (timeout: {timeout}s, attempts: {max_attempts})")

    attempt = 0

    while True:
        attempt += 1
        try:
            poller = client.begin_analyze_document(model_id, request, **kwargs)

            start_time = time.time()
            polling_interval = 4.0

            while not poller.done():
                elapsed_time = time.time() - start_time
                if elapsed_time >= timeout:
                    try:
                        poller.cancel()
                    except Exception:
                        pass
                    raise TimeoutError(f"Analysis timed out after {timeout}s")

                status = poller.status()
                print(f"[ARU] Analysis Status - {status} (elapsed: {elapsed_time:.1f}s)")

                remaining_time = min(polling_interval, max(0.0, timeout - elapsed_time))
                if remaining_time > 0:
                    poller.wait(timeout=remaining_time)

            result = poller.result()
            print("[ARU] Analysis completed")
            return result
        except Exception as exc:
            transient = _is_http_transient(exc)
            if not transient:
                print(
                    f"[ARU] Azure analysis failed (attempt {attempt}/{max_attempts}). Transient={transient}. Error: {exc}"
                )
                # Surface the provider message (e.g., inner error like: The file is corrupted or format is unsupported...)
                user_message = extract_provider_error_message(exc)
                user_message = (
                    user_message
                    + " Or try using another OCR model such as textract, gemini, or dots-ocr."
                )
                raise RequestException(message=user_message)
            if attempt >= max_attempts:
                print(
                    f"[ARU] Azure analysis failed after retries (attempt {attempt}/{max_attempts}). Transient={transient}. Error: {exc}"
                )
                raise RequestException(
                    message="OCR service is temporarily unavailable. Please retry later."
                )

            # Exponential backoff with jitter
            delay = min(max_delay, base_delay * (2 ** (attempt - 1)))
            delay = delay * (0.5 + random.random())  # 0.5x–1.5x jitter
            print(
                f"[ARU] Transient error, retrying in {delay:.1f}s (attempt {attempt}/{max_attempts}). Error: {type(exc).__name__}"
            )
            time.sleep(delay)
