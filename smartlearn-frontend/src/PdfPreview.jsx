import { API, CHAT_ID } from "./api.js";

function getDocumentFileURL(page = 1) {
  return `${API}/documents/${encodeURIComponent(CHAT_ID)}/file#page=${page}`;
}

export default function PdfPreview({ upload, activePage, previewKey }) {
  if (!upload) {
    return (
      <section className="pdf-preview">
        <div className="pdf-placeholder">
          <p>No PDF loaded</p>
          <p className="hint">Upload a PDF to see it here.</p>
        </div>
      </section>
    );
  }

  return (
    <section className="pdf-preview">
      <p className="pdf-page-label">
        Page {activePage}
      </p>
      <iframe
        key={`${previewKey}-${activePage}`}
        src={getDocumentFileURL(activePage)}
        className="pdf-frame"
        title="PDF preview"
      />
    </section>
  );
}
