"""LLM air-quality advisor — answers natural-language questions grounded in our
real forecasts via tool use (Claude).

This is the "wow" layer: a user asks *"I have asthma — is it safe to jog in
Lahore tomorrow?"* and the model calls our ``get_forecast`` tool, reads the real
number, and gives grounded health advice. The same tools are exposed over MCP
(``aqi.mcp.server``) so the system also plugs into Claude Desktop.

Needs ANTHROPIC_API_KEY in the environment. Model defaults to Claude Opus 5
(override with AQI_ADVISOR_MODEL).
"""
from __future__ import annotations

import json
import os

from aqi.tools import TOOL_SCHEMAS, run_tool
from aqi.utils.logging import get_logger

logger = get_logger("advisor")

MODEL = os.getenv("AQI_ADVISOR_MODEL", "claude-opus-5")
MAX_TOOL_ROUNDS = 6

SYSTEM_PROMPT = (
    "You are the air-quality health advisor for the Pearls AQI Predictor, which "
    "forecasts the Air Quality Index for cities across Pakistan. "
    "Ground every AQI number in real data by calling the tools — never invent "
    "values. For questions about upcoming conditions or whether it's safe to be "
    "outside, call get_forecast; for typical or seasonal air quality, call "
    "get_history_summary. Give practical, concise health guidance: who is at risk "
    "(children, elderly, respiratory/heart conditions), whether to limit outdoor "
    "exertion, mask advice, and the cleanest time of day when relevant. Keep "
    "answers short and actionable."
)


def answer(question: str, history: list[dict] | None = None) -> dict:
    """Answer a question, using tools as needed. Returns {answer, stop_reason}."""
    import anthropic

    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY from the environment
    messages: list[dict] = list(history or []) + [{"role": "user", "content": question}]

    for _ in range(MAX_TOOL_ROUNDS):
        resp = client.messages.create(
            model=MODEL,
            max_tokens=2048,
            system=SYSTEM_PROMPT,
            tools=TOOL_SCHEMAS,
            output_config={"effort": "low"},  # keep the advisor snappy; tools do the work
            messages=messages,
        )

        if resp.stop_reason == "refusal":
            return {"answer": "I can't help with that request.", "stop_reason": "refusal"}

        if resp.stop_reason == "tool_use":
            messages.append({"role": "assistant", "content": resp.content})
            tool_results = []
            for block in resp.content:
                if block.type == "tool_use":
                    result = run_tool(block.name, dict(block.input))
                    logger.info("tool %s(%s)", block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": json.dumps(result, default=str),
                    })
            messages.append({"role": "user", "content": tool_results})
            continue

        # end_turn (or any other terminal reason): return the assistant text
        text = "".join(b.text for b in resp.content if b.type == "text")
        return {"answer": text.strip(), "stop_reason": resp.stop_reason}

    return {"answer": "Sorry — I couldn't complete that. Please rephrase.", "stop_reason": "max_rounds"}


if __name__ == "__main__":
    import sys

    q = " ".join(sys.argv[1:]) or "I have asthma. Is it safe to jog in Lahore tomorrow morning?"
    print(answer(q)["answer"])
