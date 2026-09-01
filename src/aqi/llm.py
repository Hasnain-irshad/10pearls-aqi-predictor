"""Provider-agnostic chat-with-tools for the air-quality advisor.

The advisor's value is that it is *grounded*: it must answer from the system's
own forecasts by calling the functions in ``aqi.tools`` rather than from a
model's parametric knowledge. Every provider here therefore has to support
function/tool calling; the rest is request shaping.

Three providers are implemented, all over plain HTTPS with ``requests``:

======  ==================================  ==========================
Name    Environment variable                Default model
======  ==================================  ==========================
gemini  GEMINI_API_KEY / GOOGLE_API_KEY     gemini-3.6-flash
groq    GROQ_API_KEY                        llama-3.3-70b-versatile
claude  ANTHROPIC_API_KEY                   claude-sonnet-4-5
======  ==================================  ==========================

Gemini and Groq both have free tiers, which is why they are the defaults.
No vendor SDK is imported: the deployed image already depends on ``requests``,
and adding three SDKs to keep one endpoint working would undo the point of the
slim image.

Selection order: ``AQI_ADVISOR_PROVIDER`` if set, otherwise the first provider
in ``PROVIDER_ORDER`` whose key is present. Override the model with
``AQI_ADVISOR_MODEL``.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from typing import Any, Callable

from aqi.utils.logging import get_logger

logger = get_logger("llm")

REQUEST_TIMEOUT = float(os.getenv("AQI_ADVISOR_TIMEOUT", "60"))
MAX_TOKENS = int(os.getenv("AQI_ADVISOR_MAX_TOKENS", "1024"))

# Free tiers first: the advisor should work without a paid account.
PROVIDER_ORDER = ("gemini", "groq", "claude")


@dataclass
class ToolCall:
    """One function call the model asked for."""

    name: str
    args: dict
    id: str = ""


@dataclass
class Turn:
    """One model response, normalised across providers."""

    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop_reason: str = ""
    raw: Any = None


class ProviderError(RuntimeError):
    """A provider was asked to answer and could not."""


# --------------------------------------------------------------------------- #
#  Schema translation
# --------------------------------------------------------------------------- #
def _clean_schema(schema: dict) -> dict:
    """Strip keys the stricter providers reject from a JSON schema."""
    out = {k: v for k, v in schema.items() if k != "additionalProperties"}
    if "properties" in out:
        out["properties"] = {
            k: _clean_schema(v) if isinstance(v, dict) else v
            for k, v in out["properties"].items()
        }
    return out


# --------------------------------------------------------------------------- #
#  Providers
# --------------------------------------------------------------------------- #
class _Provider:
    name: str = ""
    env_keys: tuple[str, ...] = ()
    default_model: str = ""

    def __init__(self) -> None:
        # requested model -> the one the provider told us to use instead.
        self._resolved: dict[str, str] = {}

    def effective_model(self, model: str) -> str:
        """The model actually used, after any provider-suggested substitution."""
        return self._resolved.get(model, model)

    def list_models(self) -> list[str]:
        """Model ids this key can use. Used by /api/advisor/models."""
        raise NotImplementedError(f"{self.name} cannot list models.")

    def api_key(self) -> str | None:
        for key in self.env_keys:
            value = os.getenv(key)
            if value and value.strip():
                return value.strip()
        return None

    def configured(self) -> bool:
        return self.api_key() is not None

    # Each provider implements the four hooks below.
    def build_messages(self, history: list[dict], question: str) -> list:
        raise NotImplementedError

    def request(self, messages: list, tools: list[dict], system: str, model: str) -> Turn:
        raise NotImplementedError

    def record_assistant(self, messages: list, turn: Turn) -> None:
        raise NotImplementedError

    def record_tool_results(self, messages: list, turn: Turn, results: list[Any]) -> None:
        raise NotImplementedError


class Gemini(_Provider):
    name = "gemini"
    env_keys = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    default_model = "gemini-3.6-flash"
    base_url = "https://generativelanguage.googleapis.com/v1beta/models"

    # Google retires model ids and names the replacement in the 404 body, e.g.
    # "This model models/gemini-2.0-flash is no longer available. Please update
    #  your code to use models/gemini-3.6-flash". Rather than hard-fail on a
    # rotation, learn the substitution once and carry on.
    _REPLACEMENT = re.compile(r"use\s+models/([A-Za-z0-9._-]+)")

    def list_models(self) -> list[str]:
        import requests

        r = requests.get(self.base_url, params={"key": self.api_key(), "pageSize": 200},
                         timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            raise ProviderError(f"Gemini returned {r.status_code}: {r.text[:300]}")
        out = []
        for m in r.json().get("models", []):
            if "generateContent" in (m.get("supportedGenerationMethods") or ["generateContent"]):
                out.append(m.get("name", "").removeprefix("models/"))
        return sorted(n for n in out if n)

    def _declarations(self, tools: list[dict]) -> list[dict]:
        decls = []
        for t in tools:
            decl = {"name": t["name"], "description": t["description"]}
            schema = _clean_schema(t.get("input_schema", {}))
            # Gemini rejects an empty parameter object; a no-argument tool must
            # simply omit `parameters`.
            if schema.get("properties"):
                decl["parameters"] = schema
            decls.append(decl)
        return decls

    def build_messages(self, history, question):
        contents = []
        for m in history or []:
            role = "model" if m.get("role") in ("assistant", "model") else "user"
            text = m.get("content") or m.get("text") or ""
            if isinstance(text, str) and text:
                contents.append({"role": role, "parts": [{"text": text}]})
        contents.append({"role": "user", "parts": [{"text": question}]})
        return contents

    def request(self, messages, tools, system, model):
        import requests

        body = {
            "contents": messages,
            "systemInstruction": {"parts": [{"text": system}]},
            "generationConfig": {"temperature": 0.2, "maxOutputTokens": MAX_TOKENS},
        }
        if tools:
            body["tools"] = [{"functionDeclarations": self._declarations(tools)}]

        wanted = self.effective_model(model)
        r = requests.post(f"{self.base_url}/{wanted}:generateContent",
                          params={"key": self.api_key()}, json=body,
                          timeout=REQUEST_TIMEOUT)

        # A retired model id: adopt the replacement Google names and retry once.
        if r.status_code == 404:
            match = self._REPLACEMENT.search(r.text or "")
            if match and match.group(1) != wanted:
                replacement = match.group(1)
                logger.warning("Gemini model %s is retired; switching to %s.",
                               wanted, replacement)
                self._resolved[model] = replacement
                r = requests.post(f"{self.base_url}/{replacement}:generateContent",
                                  params={"key": self.api_key()}, json=body,
                                  timeout=REQUEST_TIMEOUT)

        if r.status_code >= 400:
            raise ProviderError(f"Gemini returned {r.status_code}: {r.text[:300]}")
        data = r.json()

        candidates = data.get("candidates") or []
        if not candidates:
            blocked = (data.get("promptFeedback") or {}).get("blockReason")
            raise ProviderError(f"Gemini returned no candidate ({blocked or 'unknown'}).")

        cand = candidates[0]
        parts = (cand.get("content") or {}).get("parts") or []
        text = "".join(p["text"] for p in parts if "text" in p)
        calls = [
            ToolCall(name=p["functionCall"].get("name", ""),
                     args=dict(p["functionCall"].get("args") or {}))
            for p in parts if "functionCall" in p
        ]
        return Turn(text=text, tool_calls=calls,
                    stop_reason=cand.get("finishReason", ""), raw=cand)

    def record_assistant(self, messages, turn):
        content = (turn.raw or {}).get("content")
        messages.append(content or {"role": "model", "parts": [{"text": turn.text}]})

    def record_tool_results(self, messages, turn, results):
        messages.append({
            "role": "user",
            "parts": [
                {"functionResponse": {"name": call.name, "response": {"result": result}}}
                for call, result in zip(turn.tool_calls, results)
            ],
        })


class Groq(_Provider):
    """Groq's OpenAI-compatible endpoint (also works for any OpenAI-shaped API)."""

    name = "groq"
    env_keys = ("GROQ_API_KEY",)
    default_model = "llama-3.3-70b-versatile"

    @property
    def base_url(self) -> str:
        return os.getenv("AQI_ADVISOR_BASE_URL", "https://api.groq.com/openai/v1").rstrip("/")

    def list_models(self) -> list[str]:
        import requests

        r = requests.get(f"{self.base_url}/models",
                         headers={"Authorization": f"Bearer {self.api_key()}"},
                         timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            raise ProviderError(f"Groq returned {r.status_code}: {r.text[:300]}")
        return sorted(m.get("id", "") for m in r.json().get("data", []) if m.get("id"))

    def _tools(self, tools: list[dict]) -> list[dict]:
        return [
            {"type": "function",
             "function": {"name": t["name"],
                          "description": t["description"],
                          "parameters": _clean_schema(t.get("input_schema", {}))}}
            for t in tools
        ]

    def build_messages(self, history, question):
        messages = []
        for m in history or []:
            role = "assistant" if m.get("role") in ("assistant", "model") else "user"
            content = m.get("content") or m.get("text") or ""
            if isinstance(content, str) and content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": question})
        return messages

    def request(self, messages, tools, system, model):
        import requests

        payload = {
            "model": model,
            "messages": [{"role": "system", "content": system}] + messages,
            "max_tokens": MAX_TOKENS,
            "temperature": 0.2,
        }
        if tools:
            payload["tools"] = self._tools(tools)
            payload["tool_choice"] = "auto"

        r = requests.post(f"{self.base_url}/chat/completions",
                          headers={"Authorization": f"Bearer {self.api_key()}"},
                          json=payload, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            raise ProviderError(f"Groq returned {r.status_code}: {r.text[:300]}")
        data = r.json()

        choices = data.get("choices") or []
        if not choices:
            raise ProviderError("Groq returned no choice.")
        choice = choices[0]
        message = choice.get("message") or {}

        calls = []
        for tc in message.get("tool_calls") or []:
            fn = tc.get("function") or {}
            raw_args = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw_args) if isinstance(raw_args, str) else dict(raw_args)
            except json.JSONDecodeError:
                logger.warning("Could not parse tool arguments %r; treating as empty.", raw_args)
                args = {}
            calls.append(ToolCall(name=fn.get("name", ""), args=args, id=tc.get("id", "")))

        return Turn(text=message.get("content") or "", tool_calls=calls,
                    stop_reason=choice.get("finish_reason", ""), raw=message)

    def record_assistant(self, messages, turn):
        messages.append(turn.raw or {"role": "assistant", "content": turn.text})

    def record_tool_results(self, messages, turn, results):
        for call, result in zip(turn.tool_calls, results):
            messages.append({"role": "tool", "tool_call_id": call.id,
                             "name": call.name,
                             "content": json.dumps(result, default=str)})


class Claude(_Provider):
    name = "claude"
    env_keys = ("ANTHROPIC_API_KEY",)
    default_model = "claude-sonnet-4-5"
    base_url = "https://api.anthropic.com/v1"

    def list_models(self) -> list[str]:
        import requests

        r = requests.get(f"{self.base_url}/models",
                         headers={"x-api-key": self.api_key() or "",
                                  "anthropic-version": "2023-06-01"},
                         timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            raise ProviderError(f"Anthropic returned {r.status_code}: {r.text[:300]}")
        return sorted(m.get("id", "") for m in r.json().get("data", []) if m.get("id"))

    def build_messages(self, history, question):
        messages = []
        for m in history or []:
            role = "assistant" if m.get("role") in ("assistant", "model") else "user"
            content = m.get("content") or m.get("text") or ""
            if content:
                messages.append({"role": role, "content": content})
        messages.append({"role": "user", "content": question})
        return messages

    def request(self, messages, tools, system, model):
        import requests

        payload = {"model": model, "max_tokens": MAX_TOKENS, "system": system,
                   "messages": messages}
        if tools:
            payload["tools"] = tools

        r = requests.post(f"{self.base_url}/messages",
                          headers={"x-api-key": self.api_key() or "",
                                   "anthropic-version": "2023-06-01",
                                   "content-type": "application/json"},
                          json=payload, timeout=REQUEST_TIMEOUT)
        if r.status_code >= 400:
            raise ProviderError(f"Anthropic returned {r.status_code}: {r.text[:300]}")
        data = r.json()

        blocks = data.get("content") or []
        text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
        calls = [
            ToolCall(name=b.get("name", ""), args=dict(b.get("input") or {}), id=b.get("id", ""))
            for b in blocks if b.get("type") == "tool_use"
        ]
        return Turn(text=text, tool_calls=calls,
                    stop_reason=data.get("stop_reason", ""), raw=blocks)

    def record_assistant(self, messages, turn):
        messages.append({"role": "assistant", "content": turn.raw or turn.text})

    def record_tool_results(self, messages, turn, results):
        messages.append({
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": call.id,
                 "content": json.dumps(result, default=str)}
                for call, result in zip(turn.tool_calls, results)
            ],
        })


PROVIDERS: dict[str, _Provider] = {p.name: p for p in (Gemini(), Groq(), Claude())}


# --------------------------------------------------------------------------- #
#  Selection and the shared tool loop
# --------------------------------------------------------------------------- #
def available() -> list[str]:
    """Names of every provider that has a key configured."""
    return [name for name in PROVIDER_ORDER if PROVIDERS[name].configured()]


def resolve_provider() -> _Provider | None:
    """The provider to use, or None when no key is configured anywhere."""
    requested = (os.getenv("AQI_ADVISOR_PROVIDER") or "").strip().lower()
    if requested:
        provider = PROVIDERS.get(requested)
        if provider is None:
            raise ProviderError(
                f"Unknown AQI_ADVISOR_PROVIDER '{requested}'. "
                f"Choose one of: {', '.join(PROVIDERS)}."
            )
        if not provider.configured():
            raise ProviderError(
                f"Provider '{requested}' is selected but none of "
                f"{' / '.join(provider.env_keys)} is set."
            )
        return provider
    for name in PROVIDER_ORDER:
        if PROVIDERS[name].configured():
            return PROVIDERS[name]
    return None


def model_for(provider: _Provider) -> str:
    return (os.getenv("AQI_ADVISOR_MODEL") or "").strip() or provider.default_model


def chat_with_tools(
    question: str,
    history: list[dict] | None,
    *,
    system: str,
    tools: list[dict],
    run_tool: Callable[[str, dict], Any],
    max_rounds: int = 6,
    provider: _Provider | None = None,
) -> dict:
    """Run the observe-call-answer loop until the model stops asking for tools.

    Returns ``{answer, stop_reason, provider, model, tools_used}``.
    """
    provider = provider or resolve_provider()
    if provider is None:
        raise ProviderError("No advisor provider configured.")

    model = model_for(provider)
    messages = provider.build_messages(history or [], question)
    tools_used: list[str] = []

    for _ in range(max_rounds):
        turn = provider.request(messages, tools, system, model)

        if not turn.tool_calls:
            return {
                "answer": (turn.text or "").strip(),
                "stop_reason": turn.stop_reason,
                "provider": provider.name,
                "model": provider.effective_model(model),
                "tools_used": tools_used,
            }

        provider.record_assistant(messages, turn)
        results = []
        for call in turn.tool_calls:
            logger.info("tool %s(%s) via %s", call.name, call.args, provider.name)
            tools_used.append(call.name)
            try:
                results.append(run_tool(call.name, call.args))
            except Exception as exc:  # noqa: BLE001 - report, never crash the turn
                logger.warning("tool %s failed: %s", call.name, exc)
                results.append({"error": f"{call.name} failed: {exc}"})
        provider.record_tool_results(messages, turn, results)

    return {
        "answer": "Sorry - I couldn't complete that. Please rephrase.",
        "stop_reason": "max_rounds",
        "provider": provider.name,
        "model": provider.effective_model(model),
        "tools_used": tools_used,
    }
