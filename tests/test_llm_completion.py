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


def temperature_rejected():
    return litellm.BadRequestError(
        message="Unsupported value: 'temperature' does not support 0 with this model. Only the default (1) value is supported.",
        llm_provider="openai",
        model="gpt-x",
    )


def test_retries_without_temperature_when_the_model_rejects_it(monkeypatch):
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        if "temperature" in kwargs:
            raise temperature_rejected()
        return fake_response("ok")

    monkeypatch.setattr(utils.litellm, "completion", completion)
    assert utils.llm_completion("openai/gpt-x", "hi") == "ok"
    assert [("temperature" in call) for call in calls] == [True, False]


def test_async_retries_without_temperature_when_the_model_rejects_it(monkeypatch):
    calls = []

    async def acompletion(**kwargs):
        calls.append(kwargs)
        if "temperature" in kwargs:
            raise temperature_rejected()
        return fake_response("ok")

    monkeypatch.setattr(utils.litellm, "acompletion", acompletion)
    assert asyncio.run(utils.llm_acompletion("openai/gpt-x", "hi")) == "ok"
    assert [("temperature" in call) for call in calls] == [True, False]


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
