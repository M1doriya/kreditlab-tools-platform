# SPDX-License-Identifier: Apache-2.0
"""Tests for async streaming functions in providers/model_provider_utils.py."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_image(w=10, h=10):
    return Image.new("RGB", (w, h), color=(128, 128, 128))


def _make_gemini_chunk(text=None, prompt_tokens=0, candidate_tokens=0, total_tokens=0):
    """Fake Gemini streaming chunk."""

    def _text_prop():
        if text is None:
            raise ValueError("no text")
        return text

    chunk = MagicMock()
    # Make chunk.text behave like a property that may raise ValueError
    type(chunk).text = property(lambda self: _text_prop())
    usage = SimpleNamespace(
        prompt_token_count=prompt_tokens,
        candidates_token_count=candidate_tokens,
        total_token_count=total_tokens,
    )
    chunk.usage_metadata = usage
    chunk.candidates = []
    return chunk


def _make_oai_chunk(content="", usage=None):
    """Fake OpenAI streaming chunk."""
    chunk = MagicMock()
    delta = MagicMock()
    delta.content = content
    choice = MagicMock()
    choice.delta = delta
    chunk.choices = [choice]
    chunk.usage = usage
    return chunk


async def _async_gen(*items):
    """Yield items as an async generator (used to fake stream_with_timeout)."""
    for item in items:
        yield item


# ---------------------------------------------------------------------------
# _make_gemini_call
# ---------------------------------------------------------------------------


def test_make_gemini_call_happy_path():
    """Successful Gemini streaming call returns (text, input_tokens, output_tokens)."""
    from tensorlake_docai.providers.model_provider_utils import _make_gemini_call

    mock_client = AsyncMock()
    mock_stream = AsyncMock()
    mock_client.models.generate_content_stream = AsyncMock(return_value=mock_stream)
    mock_stream.aclose = AsyncMock()

    chunk1 = _make_gemini_chunk(text="Hello ")
    chunk2 = _make_gemini_chunk(
        text="world", total_tokens=100, prompt_tokens=60, candidate_tokens=40
    )

    async def run():
        with (
            patch(
                "tensorlake_docai.providers.model_provider_utils.get_gemini_async_client_and_model",
                return_value=(mock_client, "gemini-flash"),
            ),
            patch(
                "tensorlake_docai.providers.model_provider_utils.stream_with_timeout",
                side_effect=lambda s, **kw: _async_gen(chunk1, chunk2),
            ),
            patch("google.genai.types") as mock_types,
        ):
            mock_types.ThinkingConfig.return_value = MagicMock()
            mock_types.ThinkingLevel.LOW = "LOW"
            mock_types.ThinkingLevel.MEDIUM = "MEDIUM"
            mock_types.MediaResolution.MEDIA_RESOLUTION_HIGH = "HIGH"
            mock_types.Part.from_bytes.return_value = MagicMock()

            text, inp, out = await _make_gemini_call(
                user_prompt="describe this",
                images=[_make_image()],
                job_type="vlm",
            )
        return text, inp, out

    text, inp, out = asyncio.run(run())
    assert "Hello" in text
    assert "world" in text
    assert isinstance(inp, int)
    assert isinstance(out, int)


def test_make_gemini_call_no_chunks_raises():
    """Zero chunks → raises an exception (Gemini returned no response)."""
    from tensorlake_docai.providers.model_provider_utils import _make_gemini_call

    mock_client = AsyncMock()
    mock_stream = AsyncMock()
    mock_client.models.generate_content_stream = AsyncMock(return_value=mock_stream)
    mock_stream.aclose = AsyncMock()

    async def run():
        with (
            patch(
                "tensorlake_docai.providers.model_provider_utils.get_gemini_async_client_and_model",
                return_value=(mock_client, "gemini-flash"),
            ),
            patch(
                "tensorlake_docai.providers.model_provider_utils.stream_with_timeout",
                side_effect=lambda s, **kw: _async_gen(),  # empty
            ),
            patch("google.genai.types") as mock_types,
        ):
            mock_types.ThinkingConfig.return_value = MagicMock()
            mock_types.ThinkingLevel.LOW = "LOW"
            mock_types.ThinkingLevel.MEDIUM = "MEDIUM"
            mock_types.MediaResolution.MEDIA_RESOLUTION_HIGH = "HIGH"
            mock_types.Part.from_bytes.return_value = MagicMock()

            await _make_gemini_call(
                user_prompt="test",
                images=[],
                job_type="ocr",
            )

    with pytest.raises(Exception, match="no response chunks"):
        asyncio.run(run())


def test_make_gemini_call_timeout_raises_request_exception():
    """asyncio.TimeoutError from stream_with_timeout is wrapped in RequestException."""
    from tensorlake_docai.providers.model_provider_utils import _make_gemini_call
    from tensorlake.applications import RequestError as RequestException

    mock_client = AsyncMock()
    mock_stream = AsyncMock()
    mock_client.models.generate_content_stream = AsyncMock(return_value=mock_stream)
    mock_stream.aclose = AsyncMock()

    async def _timeout_gen(s, **kw):
        raise asyncio.TimeoutError("chunk timeout")
        yield  # make it a generator

    async def run():
        with (
            patch(
                "tensorlake_docai.providers.model_provider_utils.get_gemini_async_client_and_model",
                return_value=(mock_client, "gemini-flash"),
            ),
            patch(
                "tensorlake_docai.providers.model_provider_utils.stream_with_timeout",
                side_effect=_timeout_gen,
            ),
            patch("google.genai.types") as mock_types,
        ):
            mock_types.ThinkingConfig.return_value = MagicMock()
            mock_types.ThinkingLevel.LOW = "LOW"
            mock_types.ThinkingLevel.MEDIUM = "MEDIUM"
            mock_types.MediaResolution.MEDIA_RESOLUTION_HIGH = "HIGH"
            mock_types.Part.from_bytes.return_value = MagicMock()

            await _make_gemini_call("test", images=[], job_type="ocr")

    with pytest.raises(RequestException):
        asyncio.run(run())


def test_make_gemini_call_with_pdf_bytes():
    """pdf_bytes path constructs PDF contents instead of image contents."""
    from tensorlake_docai.providers.model_provider_utils import _make_gemini_call

    mock_client = AsyncMock()
    mock_stream = AsyncMock()
    mock_client.models.generate_content_stream = AsyncMock(return_value=mock_stream)
    mock_stream.aclose = AsyncMock()

    chunk = _make_gemini_chunk(text="result", total_tokens=10, prompt_tokens=7, candidate_tokens=3)

    async def run():
        with (
            patch(
                "tensorlake_docai.providers.model_provider_utils.get_gemini_async_client_and_model",
                return_value=(mock_client, "gemini-flash"),
            ),
            patch(
                "tensorlake_docai.providers.model_provider_utils.stream_with_timeout",
                side_effect=lambda s, **kw: _async_gen(chunk),
            ),
            patch("google.genai.types") as mock_types,
        ):
            mock_types.ThinkingConfig.return_value = MagicMock()
            mock_types.ThinkingLevel.LOW = "LOW"
            mock_types.ThinkingLevel.MEDIUM = "MEDIUM"
            mock_types.MediaResolution.MEDIA_RESOLUTION_MEDIUM = "MEDIUM"
            mock_types.Part.from_bytes.return_value = MagicMock()

            return await _make_gemini_call(
                "describe", images=[], pdf_bytes=b"%PDF-1.4 fake", job_type="vlm"
            )

    text, _, _ = asyncio.run(run())
    assert text == "result"


# ---------------------------------------------------------------------------
# _make_anthropic_call
# ---------------------------------------------------------------------------


def test_make_anthropic_call_happy_path():
    """_make_anthropic_call returns (text, input_tokens, output_tokens)."""
    from tensorlake_docai.providers.model_provider_utils import _make_anthropic_call

    fake_response = SimpleNamespace(
        content=[SimpleNamespace(text="Anthropic answer")],
        usage=SimpleNamespace(input_tokens=50, output_tokens=20),
    )

    async def run():
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=fake_response)

            return await _make_anthropic_call(
                user_prompt="describe",
                images=[_make_image()],
            )

    text, inp, out = asyncio.run(run())
    assert text == "Anthropic answer"
    assert inp == 50
    assert out == 20


def test_make_anthropic_call_with_page_image():
    """page_image gets prepended to content."""
    from tensorlake_docai.providers.model_provider_utils import _make_anthropic_call

    fake_response = SimpleNamespace(
        content=[SimpleNamespace(text="ok")],
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )

    async def run():
        with patch("anthropic.AsyncAnthropic") as mock_cls:
            mock_client = AsyncMock()
            mock_cls.return_value = mock_client
            mock_client.messages.create = AsyncMock(return_value=fake_response)

            return await _make_anthropic_call(
                user_prompt="test",
                images=[_make_image()],
                page_image=_make_image(),
            )

    text, _, _ = asyncio.run(run())
    assert text == "ok"


# ---------------------------------------------------------------------------
# _make_oai_call
# ---------------------------------------------------------------------------


def test_make_oai_call_happy_path():
    """_make_oai_call streams chunks and returns accumulated text."""
    from tensorlake_docai.providers.model_provider_utils import _make_oai_call

    chunks = [_make_oai_chunk("Hello "), _make_oai_chunk("world")]

    # Build a mock async context manager client
    mock_oai_client = AsyncMock()
    mock_oai_client.chat.completions.create = AsyncMock(return_value=AsyncMock(aclose=AsyncMock()))

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_oai_client)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    async def run():
        with (
            patch(
                "tensorlake_docai.providers.model_provider_utils.get_openai_client_and_model",
                return_value=(mock_ctx, "gpt-4o"),
            ),
            patch(
                "tensorlake_docai.providers.model_provider_utils.stream_with_timeout",
                side_effect=lambda s, **kw: _async_gen(*chunks),
            ),
        ):
            return await _make_oai_call(
                user_prompt="describe",
                images=[_make_image()],
                job_type="ocr",
            )

    text, inp, out = asyncio.run(run())
    assert "Hello" in text
    assert "world" in text


def test_make_oai_call_skips_zero_size_image():
    """Images with 0 dimension are skipped silently."""
    from tensorlake_docai.providers.model_provider_utils import _make_oai_call

    zero_img = Image.new("RGB", (0, 0))
    chunks = [_make_oai_chunk("ok")]

    mock_oai_client = AsyncMock()
    mock_oai_client.chat.completions.create = AsyncMock(return_value=AsyncMock(aclose=AsyncMock()))

    mock_ctx = MagicMock()
    mock_ctx.__aenter__ = AsyncMock(return_value=mock_oai_client)
    mock_ctx.__aexit__ = AsyncMock(return_value=False)

    async def run():
        with (
            patch(
                "tensorlake_docai.providers.model_provider_utils.get_openai_client_and_model",
                return_value=(mock_ctx, "gpt-4o"),
            ),
            patch(
                "tensorlake_docai.providers.model_provider_utils.stream_with_timeout",
                side_effect=lambda s, **kw: _async_gen(*chunks),
            ),
        ):
            return await _make_oai_call("test", images=[zero_img], job_type="ocr")

    text, _, _ = asyncio.run(run())
    assert text == "ok"


# ---------------------------------------------------------------------------
# run_clients
# ---------------------------------------------------------------------------


def test_run_clients_calls_first_model_for_structured_extraction():
    """run_clients picks models[0] for structured_extraction job_type."""
    from tensorlake_docai.providers.model_provider_utils import run_clients

    async def mock_model_0(prompt, images, page_image, json_schema, job_type, **kw):
        return ("from_model_0", 10, 5)

    async def mock_model_1(prompt, images, page_image, json_schema, job_type, **kw):
        return ("from_model_1", 10, 5)

    async def run():
        with patch("asyncio.sleep", new_callable=AsyncMock):
            return await run_clients(
                user_prompt="test",
                images=[],
                models=[mock_model_0, mock_model_1],
                job_type="structured_extraction",
            )

    text, _, _ = asyncio.run(run())
    assert text == "from_model_0"


def test_run_clients_calls_second_model_for_other_job_types():
    """run_clients picks models[1] for non-structured_extraction job types."""
    from tensorlake_docai.providers.model_provider_utils import run_clients

    async def mock_model_0(prompt, images, page_image, json_schema, job_type, **kw):
        return ("from_model_0", 0, 0)

    async def mock_model_1(prompt, images, page_image, json_schema, job_type, **kw):
        return ("from_model_1", 0, 0)

    async def run():
        with patch("asyncio.sleep", new_callable=AsyncMock):
            return await run_clients(
                user_prompt="test",
                images=[],
                models=[mock_model_0, mock_model_1],
                job_type="ocr",
            )

    text, _, _ = asyncio.run(run())
    assert text == "from_model_1"


def test_run_clients_retries_on_request_exception():
    """Transient RequestException triggers retry; succeeds on second attempt."""
    from tensorlake_docai.providers.model_provider_utils import run_clients
    from tensorlake.applications import RequestError as RequestException

    call_count = 0

    async def flaky_model(prompt, images, page_image, json_schema, job_type, **kw):
        nonlocal call_count
        call_count += 1
        if call_count < 2:
            raise RequestException(message="transient error")
        return ("success", 0, 0)

    async def run():
        with patch("asyncio.sleep", new_callable=AsyncMock):
            return await run_clients(
                user_prompt="test",
                images=[],
                models=[flaky_model, flaky_model],
                job_type="structured_extraction",
            )

    text, _, _ = asyncio.run(run())
    assert text == "success"
    assert call_count == 2


def test_run_clients_schema_error_does_not_retry():
    """SCHEMA_ERROR in RequestException is not retried."""
    from tensorlake_docai.providers.model_provider_utils import run_clients
    from tensorlake.applications import RequestError as RequestException

    call_count = 0

    async def bad_schema_model(prompt, images, page_image, json_schema, job_type, **kw):
        nonlocal call_count
        call_count += 1
        raise RequestException(message="SCHEMA_ERROR: bad field")

    async def run():
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await run_clients(
                user_prompt="test",
                images=[],
                models=[bad_schema_model, bad_schema_model],
                job_type="structured_extraction",
            )

    with pytest.raises(RequestException, match="SCHEMA_ERROR"):
        asyncio.run(run())
    assert call_count == 1  # no retries


def test_run_clients_raises_after_all_retries_exhausted():
    """All 3 attempts fail → RequestException raised."""
    from tensorlake_docai.providers.model_provider_utils import run_clients
    from tensorlake.applications import RequestError as RequestException

    async def always_fails(prompt, images, page_image, json_schema, job_type, **kw):
        raise RequestException(message="permanent failure")

    async def run():
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await run_clients(
                user_prompt="test",
                images=[],
                models=[always_fails, always_fails],
                job_type="structured_extraction",
            )

    with pytest.raises(RequestException):
        asyncio.run(run())


def test_run_clients_none_output_raises():
    """Model returning None text raises RequestException."""
    from tensorlake_docai.providers.model_provider_utils import run_clients
    from tensorlake.applications import RequestError as RequestException

    async def null_model(prompt, images, page_image, json_schema, job_type, **kw):
        return (None, 0, 0)

    async def run():
        with patch("asyncio.sleep", new_callable=AsyncMock):
            await run_clients(
                user_prompt="test",
                images=[],
                models=[null_model, null_model],
                job_type="structured_extraction",
            )

    with pytest.raises(RequestException):
        asyncio.run(run())
