# SPDX-License-Identifier: Apache-2.0
"""Azure Document Intelligence retry/backoff behavior. Network calls are
mocked — these tests verify the classifier and the retry loop logic, not Azure
itself."""

from unittest.mock import MagicMock

import pytest

from azure.core.exceptions import (
    HttpResponseError,
    ServiceRequestError,
    ServiceResponseError,
)
from requests.exceptions import ConnectionError as ReqConnectionError
from requests.exceptions import ReadTimeout, SSLError

from tensorlake.applications import RequestError

from tensorlake_docai.ocr import azure_retry_utils
from tensorlake_docai.ocr.azure_retry_utils import (
    _is_http_transient,
    robust_azure_analyze_document,
)

# --- transient classifier --------------------------------------------------


def _http_err(status: int) -> HttpResponseError:
    err = HttpResponseError(message=f"status {status}")
    err.status_code = status
    return err


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_http_transient_status_codes(status):
    assert _is_http_transient(_http_err(status))


@pytest.mark.parametrize("status", [400, 401, 403, 404, 422])
def test_http_non_transient_status_codes(status):
    assert not _is_http_transient(_http_err(status))


def test_network_errors_are_transient():
    assert _is_http_transient(ServiceRequestError(message="x"))
    assert _is_http_transient(ServiceResponseError(message="x"))
    assert _is_http_transient(ReadTimeout())
    assert _is_http_transient(SSLError())
    assert _is_http_transient(ReqConnectionError())
    assert _is_http_transient(TimeoutError())


def test_unrelated_errors_not_transient():
    assert not _is_http_transient(ValueError("oops"))
    assert not _is_http_transient(KeyError("missing"))


# --- robust_azure_analyze_document ----------------------------------------


def _make_poller(result=None):
    """Builds a fake LRO poller that completes immediately with `result`."""
    poller = MagicMock()
    poller.done.return_value = True
    poller.status.return_value = "succeeded"
    poller.result.return_value = result
    return poller


def test_succeeds_first_attempt():
    client = MagicMock()
    expected = {"pages": []}
    client.begin_analyze_document.return_value = _make_poller(result=expected)

    out = robust_azure_analyze_document(client, "prebuilt-layout", request=b"file")
    assert out == expected
    assert client.begin_analyze_document.call_count == 1


def test_retries_then_succeeds_on_transient(monkeypatch):
    # No real sleeping during retries
    monkeypatch.setattr(azure_retry_utils.time, "sleep", lambda _s: None)

    expected = {"ok": True}
    client = MagicMock()
    # MagicMock.side_effect treats callables (like a poller MagicMock) as returns,
    # so we mix exception instances with the poller directly.
    client.begin_analyze_document.side_effect = [
        ServiceRequestError(message="reset"),
        _http_err(503),
        _make_poller(result=expected),
    ]

    out = robust_azure_analyze_document(client, "prebuilt-layout", request=b"file")
    assert out == expected
    assert client.begin_analyze_document.call_count == 3


def test_non_transient_raises_request_error_immediately(monkeypatch):
    monkeypatch.setattr(azure_retry_utils.time, "sleep", lambda _s: None)

    client = MagicMock()
    client.begin_analyze_document.side_effect = _http_err(400)

    with pytest.raises(RequestError):
        robust_azure_analyze_document(client, "prebuilt-layout", request=b"file")
    # No retries on non-transient
    assert client.begin_analyze_document.call_count == 1


def test_gives_up_after_max_attempts(monkeypatch):
    monkeypatch.setattr(azure_retry_utils.time, "sleep", lambda _s: None)

    client = MagicMock()
    # Always raises a transient error
    client.begin_analyze_document.side_effect = ServiceRequestError(message="boom")

    with pytest.raises(RequestError) as excinfo:
        robust_azure_analyze_document(client, "prebuilt-layout", request=b"file")
    # Module-internal constant is 10 attempts
    assert client.begin_analyze_document.call_count == 10
    assert (
        "temporarily unavailable" in str(excinfo.value).lower()
        or "temporarily unavailable" in getattr(excinfo.value, "message", "").lower()
    )


def test_polling_timeout_raises_timeout_then_retries(monkeypatch):
    """Poller never completes → TimeoutError → treated as transient → retried."""
    monkeypatch.setattr(azure_retry_utils.time, "sleep", lambda _s: None)

    # Fake `time.time()` so the elapsed check trips immediately.
    fake_now = iter([0.0, 0.0, 9999.0, 9999.0, 9999.0, 9999.0])
    monkeypatch.setattr(azure_retry_utils.time, "time", lambda: next(fake_now, 9999.0))

    never_done_poller = MagicMock()
    never_done_poller.done.return_value = False
    never_done_poller.status.return_value = "running"
    never_done_poller.wait.return_value = None

    good_poller = _make_poller(result={"ok": True})

    client = MagicMock()
    client.begin_analyze_document.side_effect = [never_done_poller, good_poller]

    out = robust_azure_analyze_document(client, "prebuilt-layout", request=b"file", timeout=1)
    assert out == {"ok": True}
    assert client.begin_analyze_document.call_count == 2
    # Cancel was attempted on timeout
    never_done_poller.cancel.assert_called_once()
