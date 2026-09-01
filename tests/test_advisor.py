"""Tests for the pluggable advisor provider layer.

These exercise request shaping, response parsing and the tool loop against
recorded response shapes, so they need no API key and no network. What they
cannot prove is that a live provider accepts the request; that needs a key.
"""
import json

import pytest
import requests

from aqi import llm
from aqi.llm import Gemini, Groq, Claude, ProviderError, chat_with_tools

TOOLS = [
    {
        "name": "list_cities",
        "description": "List cities.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "get_forecast",
        "description": "Forecast for a city.",
        "input_schema": {
            "type": "object",
            "properties": {"city": {"type": "string", "description": "City name"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
]


class FakeResponse:
    def __init__(self, payload, status_code=200):
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload)

    def json(self):
        return self._payload


@pytest.fixture
def capture(monkeypatch):
    """Replace requests.post with a scripted sequence, recording every request."""
    sent = []

    def make(responses):
        queue = list(responses)

        def fake_post(url, **kwargs):
            sent.append({"url": url, "json": kwargs.get("json"),
                         "headers": kwargs.get("headers"), "params": kwargs.get("params")})
            return queue.pop(0)

        monkeypatch.setattr(requests, "post", fake_post)
        return sent

    return make


def _run(provider, responses, capture, monkeypatch, key_env, key_name):
    monkeypatch.setenv(key_env, "test-key")
    monkeypatch.delenv("AQI_ADVISOR_MODEL", raising=False)
    sent = capture(responses)
    calls = []

    def run_tool(name, args):
        calls.append((name, args))
        return {"city": args.get("city", "?"), "current_aqi": 153}

    result = chat_with_tools("Is it safe to jog in Lahore?", None,
                             system="SYSTEM", tools=TOOLS, run_tool=run_tool,
                             provider=provider)
    return result, sent, calls


# --------------------------------------------------------------------- Gemini
def test_gemini_tool_loop(capture, monkeypatch):
    responses = [
        FakeResponse({"candidates": [{"content": {"role": "model", "parts": [
            {"functionCall": {"name": "get_forecast", "args": {"city": "Lahore"}}}]},
            "finishReason": "STOP"}]}),
        FakeResponse({"candidates": [{"content": {"role": "model", "parts": [
            {"text": "Lahore is 153, unhealthy - limit outdoor exertion."}]},
            "finishReason": "STOP"}]}),
    ]
    result, sent, calls = _run(Gemini(), responses, capture, monkeypatch,
                               "GEMINI_API_KEY", "key")

    assert calls == [("get_forecast", {"city": "Lahore"})]
    assert "153" in result["answer"]
    assert result["provider"] == "gemini"
    assert result["tools_used"] == ["get_forecast"]

    first = sent[0]["json"]
    assert first["systemInstruction"]["parts"][0]["text"] == "SYSTEM"
    decls = first["tools"][0]["functionDeclarations"]
    # a no-argument tool must omit `parameters` entirely
    assert "parameters" not in decls[0]
    # `additionalProperties` must be stripped for Gemini
    assert "additionalProperties" not in decls[1]["parameters"]
    assert decls[1]["parameters"]["properties"]["city"]["type"] == "string"
    # the key travels as a query parameter, not a header
    assert sent[0]["params"]["key"] == "test-key"

    # the second request must carry the model turn and the function response
    second = sent[1]["json"]["contents"]
    assert second[-1]["parts"][0]["functionResponse"]["name"] == "get_forecast"
    assert second[-1]["parts"][0]["functionResponse"]["response"]["result"]["current_aqi"] == 153


def test_gemini_reports_a_blocked_prompt(capture, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    capture([FakeResponse({"promptFeedback": {"blockReason": "SAFETY"}})])
    with pytest.raises(ProviderError, match="SAFETY"):
        chat_with_tools("q", None, system="S", tools=TOOLS,
                        run_tool=lambda *a: {}, provider=Gemini())


# ----------------------------------------------------------------------- Groq
def test_groq_tool_loop(capture, monkeypatch):
    responses = [
        FakeResponse({"choices": [{"message": {"role": "assistant", "content": None,
            "tool_calls": [{"id": "call_1", "type": "function", "function": {
                "name": "get_forecast", "arguments": '{"city": "Lahore"}'}}]},
            "finish_reason": "tool_calls"}]}),
        FakeResponse({"choices": [{"message": {"role": "assistant",
            "content": "Lahore is 153 - unhealthy."}, "finish_reason": "stop"}]}),
    ]
    result, sent, calls = _run(Groq(), responses, capture, monkeypatch,
                               "GROQ_API_KEY", "key")

    assert calls == [("get_forecast", {"city": "Lahore"})]
    assert "153" in result["answer"]
    assert result["provider"] == "groq"

    first = sent[0]["json"]
    assert first["messages"][0] == {"role": "system", "content": "SYSTEM"}
    assert first["tools"][0]["type"] == "function"
    assert first["tools"][0]["function"]["name"] == "list_cities"
    assert first["tool_choice"] == "auto"
    assert sent[0]["headers"]["Authorization"] == "Bearer test-key"

    # tool results go back as role=tool keyed by tool_call_id
    tool_msg = sent[1]["json"]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "call_1"
    assert json.loads(tool_msg["content"])["current_aqi"] == 153


def test_groq_survives_unparsable_tool_arguments(capture, monkeypatch):
    responses = [
        FakeResponse({"choices": [{"message": {"tool_calls": [
            {"id": "c1", "function": {"name": "list_cities", "arguments": "not json"}}]},
            "finish_reason": "tool_calls"}]}),
        FakeResponse({"choices": [{"message": {"content": "done"}, "finish_reason": "stop"}]}),
    ]
    result, _sent, calls = _run(Groq(), responses, capture, monkeypatch,
                                "GROQ_API_KEY", "key")
    assert calls == [("list_cities", {})]     # degraded to empty args, did not crash
    assert result["answer"] == "done"


# --------------------------------------------------------------------- Claude
def test_claude_tool_loop(capture, monkeypatch):
    responses = [
        FakeResponse({"content": [{"type": "tool_use", "id": "tu_1",
                                   "name": "get_forecast", "input": {"city": "Lahore"}}],
                      "stop_reason": "tool_use"}),
        FakeResponse({"content": [{"type": "text", "text": "Lahore is 153."}],
                      "stop_reason": "end_turn"}),
    ]
    result, sent, calls = _run(Claude(), responses, capture, monkeypatch,
                               "ANTHROPIC_API_KEY", "key")

    assert calls == [("get_forecast", {"city": "Lahore"})]
    assert result["answer"] == "Lahore is 153."
    assert sent[0]["headers"]["x-api-key"] == "test-key"
    assert sent[0]["headers"]["anthropic-version"] == "2023-06-01"
    assert sent[1]["json"]["messages"][-1]["content"][0]["type"] == "tool_result"


# ------------------------------------------------------------------ selection
def test_free_providers_are_preferred(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.delenv("AQI_ADVISOR_PROVIDER", raising=False)

    assert llm.resolve_provider() is None            # nothing configured

    monkeypatch.setenv("ANTHROPIC_API_KEY", "x")
    assert llm.resolve_provider().name == "claude"

    monkeypatch.setenv("GROQ_API_KEY", "x")
    assert llm.resolve_provider().name == "groq"     # free tier wins

    monkeypatch.setenv("GEMINI_API_KEY", "x")
    assert llm.resolve_provider().name == "gemini"   # first in the order
    assert llm.available() == ["gemini", "groq", "claude"]


def test_explicit_provider_override_and_errors(monkeypatch):
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "GROQ_API_KEY", "ANTHROPIC_API_KEY"):
        monkeypatch.delenv(k, raising=False)

    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.setenv("AQI_ADVISOR_PROVIDER", "groq")
    with pytest.raises(ProviderError, match="GROQ_API_KEY"):
        llm.resolve_provider()                        # selected but unconfigured

    monkeypatch.setenv("AQI_ADVISOR_PROVIDER", "nope")
    with pytest.raises(ProviderError, match="Unknown"):
        llm.resolve_provider()

    monkeypatch.setenv("AQI_ADVISOR_PROVIDER", "gemini")
    assert llm.resolve_provider().name == "gemini"


def test_model_override(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "x")
    monkeypatch.delenv("AQI_ADVISOR_MODEL", raising=False)
    assert llm.model_for(Gemini()) == "gemini-3.6-flash"
    monkeypatch.setenv("AQI_ADVISOR_MODEL", "gemini-2.5-flash")
    assert llm.model_for(Gemini()) == "gemini-2.5-flash"


def test_http_error_is_reported_with_status(capture, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    capture([FakeResponse({"error": {"message": "rate limited"}}, status_code=429)])
    with pytest.raises(ProviderError, match="429"):
        chat_with_tools("q", None, system="S", tools=TOOLS,
                        run_tool=lambda *a: {}, provider=Groq())


def test_a_failing_tool_does_not_abort_the_conversation(capture, monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "x")
    responses = [
        FakeResponse({"choices": [{"message": {"tool_calls": [
            {"id": "c1", "function": {"name": "get_forecast", "arguments": '{"city":"X"}'}}]},
            "finish_reason": "tool_calls"}]}),
        FakeResponse({"choices": [{"message": {"content": "I could not look that up."},
                                   "finish_reason": "stop"}]}),
    ]
    sent = capture(responses)

    def boom(name, args):
        raise RuntimeError("store offline")

    result = chat_with_tools("q", None, system="S", tools=TOOLS,
                             run_tool=boom, provider=Groq())
    assert result["answer"] == "I could not look that up."
    assert "store offline" in sent[1]["json"]["messages"][-1]["content"]


# ------------------------------------------------- retired-model self-healing
def test_gemini_adopts_the_replacement_model_named_in_a_404(capture, monkeypatch):
    """Google retires ids and names the successor in the 404 body; adopt it."""
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("AQI_ADVISOR_MODEL", "gemini-2.0-flash")   # pinned to a retired id

    retired = FakeResponse({"error": {"code": 404, "message":
        "This model models/gemini-2.0-flash is no longer available. Please "
        "update your code to use models/gemini-3.6-flash for the latest "
        "features and improvements.", "status": "NOT_FOUND"}}, status_code=404)
    ok = FakeResponse({"candidates": [{"content": {"role": "model", "parts": [
        {"text": "Thursday looks cleanest."}]}, "finishReason": "STOP"}]})

    provider = Gemini()
    sent = capture([retired, ok, ok])
    result = chat_with_tools("Which day is cleanest?", None, system="S", tools=TOOLS,
                             run_tool=lambda *a: {}, provider=provider)

    assert result["answer"] == "Thursday looks cleanest."
    assert "gemini-2.0-flash:generateContent" in sent[0]["url"]   # tried the pinned id
    assert "gemini-3.6-flash:generateContent" in sent[1]["url"]   # retried on the successor
    assert result["model"] == "gemini-3.6-flash"                  # reports what was used

    # the substitution is remembered, so later turns do not repeat the 404
    provider.request([], TOOLS, "S", "gemini-2.0-flash")
    assert "gemini-3.6-flash:generateContent" in sent[2]["url"]


def test_gemini_404_without_a_suggestion_is_reported(capture, monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    capture([FakeResponse({"error": {"message": "not found"}}, status_code=404)])
    with pytest.raises(ProviderError, match="404"):
        chat_with_tools("q", None, system="S", tools=TOOLS,
                        run_tool=lambda *a: {}, provider=Gemini())


def test_default_gemini_model_is_current(monkeypatch):
    monkeypatch.delenv("AQI_ADVISOR_MODEL", raising=False)
    assert Gemini().default_model == "gemini-3.6-flash"


def test_list_models(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "k")
    monkeypatch.setenv("GROQ_API_KEY", "k")

    def fake_get(url, **kwargs):
        if "generativelanguage" in url:
            return FakeResponse({"models": [
                {"name": "models/gemini-3.6-flash",
                 "supportedGenerationMethods": ["generateContent"]},
                {"name": "models/text-embedding-004",
                 "supportedGenerationMethods": ["embedContent"]}]})
        return FakeResponse({"data": [{"id": "llama-3.3-70b-versatile"}, {"id": "whisper"}]})

    monkeypatch.setattr(requests, "get", fake_get)
    assert Gemini().list_models() == ["gemini-3.6-flash"]      # embedding model filtered out
    assert Groq().list_models() == ["llama-3.3-70b-versatile", "whisper"]
