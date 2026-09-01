"""LLM air-quality advisor - answers questions grounded in our real forecasts.

This is the "wow" layer: a user asks *"I have asthma - is it safe to jog in
Lahore tomorrow?"* and the model calls our ``get_forecast`` tool, reads the real
number, and gives grounded health advice. The same tools are exposed over MCP
(``aqi.mcp.server``), so the dashboard advisor and any external protocol client
answer from exactly the same data.

The provider is pluggable (see ``aqi.llm``) because the advisor should not
depend on a paid account. With no configuration it uses whichever of
GEMINI_API_KEY, GROQ_API_KEY or ANTHROPIC_API_KEY is present, preferring the
free tiers. Force one with AQI_ADVISOR_PROVIDER=gemini|groq|claude and override
the model with AQI_ADVISOR_MODEL.
"""
from __future__ import annotations

from aqi.llm import ProviderError, available, chat_with_tools, model_for, resolve_provider
from aqi.tools import TOOL_SCHEMAS, run_tool
from aqi.utils.logging import get_logger

logger = get_logger("advisor")

MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = (
    "You are the air-quality health advisor for the Pearls AQI Predictor, which "
    "forecasts the Air Quality Index for cities across Pakistan. "
    "Ground every AQI number in real data by calling the tools - never invent "
    "values. For questions about upcoming conditions or whether it's safe to be "
    "outside, call get_forecast; for typical or seasonal air quality, call "
    "get_history_summary. Give practical, concise health guidance: who is at risk "
    "(children, elderly, respiratory/heart conditions), whether to limit outdoor "
    "exertion, mask advice, and the cleanest time of day when relevant. Keep "
    "answers short and actionable.\n\n"
    "Format the reply as Markdown for a narrow side panel:\n"
    "- Open with one or two sentences that answer the question directly, with "
    "the key value in bold.\n"
    "- Then, only if there is more to say, a short '### Guidance' section of at "
    "most four bullets.\n"
    "- Bold every AQI number, category name and day.\n"
    "- No tables, no code blocks, no headings above level 3, and never more "
    "than about 120 words in total."
)


def is_configured() -> bool:
    """True when at least one provider has a key set."""
    return bool(available())


def status() -> dict:
    """What the advisor would use, for /api/health and for diagnostics."""
    try:
        provider = resolve_provider()
    except ProviderError as exc:
        return {"configured": False, "detail": str(exc)}
    if provider is None:
        return {
            "configured": False,
            "detail": ("No advisor key configured. Set GEMINI_API_KEY (free at "
                       "aistudio.google.com) or GROQ_API_KEY (free at console.groq.com)."),
        }
    return {
        "configured": True,
        "provider": provider.name,
        "model": model_for(provider),
        "available": available(),
    }


def answer(question: str, history: list[dict] | None = None) -> dict:
    """Answer a question, using the grounded tools as needed.

    Returns ``{answer, stop_reason, provider, model, tools_used}``.
    """
    return chat_with_tools(
        question,
        history,
        system=SYSTEM_PROMPT,
        tools=TOOL_SCHEMAS,
        run_tool=run_tool,
        max_rounds=MAX_TOOL_ROUNDS,
    )


if __name__ == "__main__":
    import sys

    print("advisor status:", status())
    q = " ".join(sys.argv[1:]) or "I have asthma. Is it safe to jog in Lahore tomorrow morning?"
    print(answer(q)["answer"])
