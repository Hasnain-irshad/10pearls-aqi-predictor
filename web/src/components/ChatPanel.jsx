import { useState, useRef, useEffect } from "react";
import { api } from "../api";
import { renderMarkdown } from "../markdown";

// AI air-quality advisor — grounded in the real model via the /api/chat backend.
export default function ChatPanel({ city }) {
  const [messages, setMessages] = useState([]); // {role, content}
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState(null);
  const endRef = useRef(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, busy]);

  const suggestions = [
    `Is it safe to exercise outdoors in ${city} today?`,
    `I have asthma — any precautions for ${city} tomorrow?`,
    `Which day this week has the cleanest air in ${city}?`,
  ];

  async function send(question) {
    const q = (question ?? input).trim();
    if (!q || busy) return;
    setError(null);
    setInput("");
    const history = messages.map((m) => ({ role: m.role, content: m.content }));
    setMessages((m) => [...m, { role: "user", content: q }]);
    setBusy(true);
    try {
      const res = await api.chat(q, history);
      setMessages((m) => [...m, { role: "assistant", content: res.answer }]);
    } catch (e) {
      setError(e.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="chat-card">
      <h3>💬 Ask the AQI advisor</h3>
      <p className="chat-hint">AI answers grounded in the live forecast (via MCP tools).</p>

      <div className="chat-log">
        {messages.length === 0 && (
          <div className="chat-suggest">
            {suggestions.map((s) => (
              <button key={s} onClick={() => send(s)} disabled={busy}>
                {s}
              </button>
            ))}
          </div>
        )}
        {messages.map((m, i) =>
          m.role === "assistant" ? (
            // The advisor replies in Markdown; render it rather than printing
            // the asterisks. renderMarkdown escapes the text before adding any
            // tags, so model output cannot inject HTML.
            <div
              key={i}
              className="chat-msg assistant md"
              dangerouslySetInnerHTML={{ __html: renderMarkdown(m.content) }}
            />
          ) : (
            <div key={i} className="chat-msg user">
              {m.content}
            </div>
          )
        )}
        {busy && <div className="chat-msg assistant thinking">…thinking</div>}
        {error && <div className="chat-error">{error}</div>}
        <div ref={endRef} />
      </div>

      <form
        className="chat-input"
        onSubmit={(e) => {
          e.preventDefault();
          send();
        }}
      >
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          placeholder={`Ask about air quality in ${city}…`}
          disabled={busy}
        />
        <button type="submit" disabled={busy || !input.trim()}>
          Send
        </button>
      </form>
    </div>
  );
}
