import asyncio
from types import SimpleNamespace

import litellm
import pytest

from pageindex import utils


def fake_response(text):
    return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=text), finish_reason="stop")])


def auth_error():
    return litellm.AuthenticationError(message="bad key", llm_provider="openai", model="gpt-x")


def test_auth_error_raises_immediately(monkeypatch):
    calls = []

    def boom(**kwargs):
        calls.append(kwargs)
        raise auth_error()

    monkeypatch.setattr(utils.litellm, "completion", boom)
    monkeypatch.setattr(utils.time, "sleep", lambda seconds: None)
    with pytest.raises(litellm.AuthenticationError):
        utils.llm_completion("openai/gpt-x", "hi")
    assert len(calls) == 1


def test_rate_limit_is_still_retried(monkeypatch):
    outcomes = iter([litellm.RateLimitError(message="slow down", llm_provider="openai", model="gpt-x"), fake_response("ok")])

    def flaky(**kwargs):
        outcome = next(outcomes)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr(utils.litellm, "completion", flaky)
    monkeypatch.setattr(utils.time, "sleep", lambda seconds: None)
    assert utils.llm_completion("openai/gpt-x", "hi") == "ok"


def test_async_auth_error_raises_immediately(monkeypatch):
    calls = []

    async def boom(**kwargs):
        calls.append(kwargs)
        raise auth_error()

    monkeypatch.setattr(utils.litellm, "acompletion", boom)
    with pytest.raises(litellm.AuthenticationError):
        asyncio.run(utils.llm_acompletion("openai/gpt-x", "hi"))
    assert len(calls) == 1


def test_completion_passes_request_timeout(monkeypatch):
    calls = []

    def fake(**kwargs):
        calls.append(kwargs)
        return fake_response("ok")

    monkeypatch.setattr(utils.litellm, "completion", fake)
    utils.llm_completion("openai/gpt-x", "hi")
    assert calls[0]["timeout"] == utils.LLM_REQUEST_TIMEOUT_SECONDS


def test_requests_leave_sampling_parameters_at_the_model_default(monkeypatch):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return fake_response("ok")

    async def acompletion(**kwargs):
        calls.append(kwargs)
        return fake_response("ok")

    monkeypatch.setattr(utils.litellm, "completion", completion)
    monkeypatch.setattr(utils.litellm, "acompletion", acompletion)
    utils.llm_completion("openai/gpt-x", "hi")
    asyncio.run(utils.llm_acompletion("openai/gpt-x", "hi", response_format={"type": "json_object"}))

    assert len(calls) == 2
    assert all("temperature" not in call for call in calls)
    assert calls[1]["response_format"] == {"type": "json_object"}


def test_other_bad_requests_still_raise_immediately(monkeypatch):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        raise litellm.BadRequestError(message="context length exceeded", llm_provider="openai", model="gpt-x")

    monkeypatch.setattr(utils.litellm, "completion", completion)
    with pytest.raises(litellm.BadRequestError):
        utils.llm_completion("openai/gpt-x", "hi")
    assert len(calls) == 1


def test_completion_passes_response_format_when_requested(monkeypatch):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        return fake_response('{"ok": true}')

    monkeypatch.setattr(utils.litellm, "completion", completion)
    utils.llm_completion("openai/gpt-x", "hi", response_format={"type": "json_object"})
    utils.llm_completion("openai/gpt-x", "hi")

    assert calls[0]["response_format"] == {"type": "json_object"}
    assert "response_format" not in calls[1]
