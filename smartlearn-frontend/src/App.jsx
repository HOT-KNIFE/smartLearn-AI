import { useState } from "react";
import { uploadPDF, askQuestion } from "./api.js";

export default function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [message, setMessage] = useState("");
  const [answer, setAnswer] = useState(null);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  async function handleUpload() {
    if (!file) return;
    try {
      setStatus("uploading");
      setError("");
      setUpload(null);
      setAnswer(null);
      const data = await uploadPDF(file);
      setUpload(data);
    } catch (err) {
      setError(err.message || "Upload failed");
    } finally {
      setStatus("idle");
    }
  }

  async function handleAsk(e) {
    e.preventDefault();
    if (!message.trim()) return;
    try {
      setStatus("asking");
      setError("");
      setAnswer(null);
      const data = await askQuestion(message.trim());
      setAnswer(data);
    } catch (err) {
      setError(err.message || "Chat failed");
    } finally {
      setStatus("idle");
    }
  }

  return (
    <main>
      <h1>SmartLearn Lite</h1>

      <section className="card">
        <label htmlFor="pdf-file">PDF file</label>
        <input
          id="pdf-file"
          type="file"
          accept="application/pdf"
          onChange={(e) => setFile(e.target.files[0])}
        />
        <button
          onClick={handleUpload}
          disabled={!file || status !== "idle"}
        >
          {status === "uploading" ? "Uploading…" : "Upload"}
        </button>
      </section>

      {upload && (
        <section className="card">
          <p>
            Uploaded: {upload.filename} ({upload.pages} pages, {upload.characters} characters)
          </p>
        </section>
      )}

      <form className="card" onSubmit={handleAsk}>
        <label htmlFor="message">Message</label>
        <textarea
          id="message"
          value={message}
          onChange={(e) => setMessage(e.target.value)}
        />
        <button
          type="submit"
          disabled={!upload || !message.trim() || status !== "idle"}
        >
          {status === "asking" ? "Asking…" : "Ask"}
        </button>
      </form>

      {status !== "idle" && <p>{status === "uploading" ? "Uploading…" : "Asking…"}</p>}

      {error && <p className="error" role="alert">{error}</p>}

      {answer && (
        <section className="card">
          <p>{answer.answer}</p>
          {answer.citations?.length > 0 && (
            <div>
              {answer.citations.map((page) => (
                <span key={page}>Page {page}</span>
              ))}
            </div>
          )}
        </section>
      )}
    </main>
  );
}
