import { useState } from "react";
import { uploadPDF } from "./api.js";
import PdfPreview from "./PdfPreview.jsx";
import ChatPanel from "./ChatPanel.jsx";

export default function App() {
  const [file, setFile] = useState(null);
  const [upload, setUpload] = useState(null);
  const [activePage, setActivePage] = useState(1);
  const [previewKey, setPreviewKey] = useState(0);
  const [status, setStatus] = useState("idle");
  const [error, setError] = useState("");

  async function handleUpload() {
    if (!file) return;
    try {
      setStatus("uploading");
      setError("");
      const data = await uploadPDF(file);
      setUpload(data);
      setActivePage(1);
      setPreviewKey((k) => k + 1);
    } catch (err) {
      setError(err.message || "Upload failed");
    } finally {
      setStatus("idle");
    }
  }

  function handleJumpToPage(page) {
    setActivePage(page);
  }

  return (
    <main>
      <h1>SmartLearn AI</h1>

      {/* Upload bar */}
      <section className="card upload-bar">
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
        {upload && (
          <span className="upload-info">
            {upload.filename} ({upload.pages} pages)
          </span>
        )}
      </section>

      {error && <p className="error" role="alert">{error}</p>}

      {/* Workspace: left = preview, right = chat */}
      <div className="workspace">
        <div className="workspace-preview">
          <PdfPreview
            upload={upload}
            activePage={activePage}
            previewKey={previewKey}
          />
        </div>
        <div className="workspace-chat">
          <ChatPanel
            key={previewKey}
            enabled={!!upload}
            disabled={!upload}
            onJumpToPage={handleJumpToPage}
          />
        </div>
      </div>
    </main>
  );
}
