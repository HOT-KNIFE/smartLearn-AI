import { useState } from "react";
import { askQuestion } from "./api.js";

export default function ChatPanel({ enabled, onBusy, disabled, onJumpToPage }) {
  const [message, setMessage] = useState("");
  const [messages, setMessages] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  async function handleAsk(e) {
    e.preventDefault();
    if (!message.trim() || loading) return;

    const userMsg = message.trim();
    setMessage("");
    setError("");
    setLoading(true);
    if (onBusy) onBusy(true);

    // Append user message immediately
    setMessages((prev) => [...prev, { role: "user", content: userMsg }]);

    try {
      const data = await askQuestion(userMsg);
      setMessages((prev) => [
        ...prev,
        {
          role: "assistant",
          content: data.answer,
          citations: data.citations || [],
          sources: data.sources || [],
        },
      ]);
    } catch (err) {
      setError(err.message || "Chat failed");
    } finally {
      setLoading(false);
      if (onBusy) onBusy(false);
    }
  }

  return (
    <section className="chat-panel">
      {/* Messages */}
      <div className="chat-messages">
        {messages.length === 0 && (
          <p className="chat-empty">
            Upload a PDF and ask a question to start.
          </p>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`chat-bubble ${msg.role}`}>
            <div className="chat-role">
              {msg.role === "user" ? "You" : "Assistant"}
            </div>
            <div className="chat-content">{msg.content}</div>
            {msg.citations?.length > 0 && (
              <div className="chat-citations">
                {msg.citations.map((page) => (
                  <button
                    key={page}
                    className="citation-btn"
                    onClick={() => onJumpToPage?.(page)}
                  >
                    Page {page}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="chat-bubble assistant">
            <div className="chat-role">Assistant</div>
            <div className="chat-content typing">Thinking...</div>
          </div>
        )}
      </div>

      {/* Error */}
      {error && (
        <p className="chat-error" role="alert">
          {error}
        </p>
      )}

      {/* Input */}
      <form className="chat-form" onSubmit={handleAsk}>
        <textarea
          id="chat-message"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
          placeholder="Ask a question about the PDF..."
          disabled={disabled || loading}
        />
        <button
          type="submit"
          disabled={!enabled || !message.trim() || loading}
        >
          {loading ? "Asking..." : "Ask"}
        </button>
      </form>
    </section>
  );
}
