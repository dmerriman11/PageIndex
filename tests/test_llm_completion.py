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
