"""RAG helpers: text cleaning, page extraction, chunking, embeddings, and artifact storage."""

import json
import re
from pathlib import Path
from io import BytesIO

from pypdf import PdfReader


# ---------------------------------------------------------------------------
# 1. Text cleaning
# ---------------------------------------------------------------------------

def clean_text(text: str) -> str:
    """Normalize extracted PDF text.

    Removes null bytes, soft hyphens, repeated whitespace, and noisy
    intra-paragraph line breaks while keeping deliberate paragraph breaks.
    """
    if not text:
        return ""

    # Remove null bytes
    text = text.replace("\x00", "")

    # Remove soft hyphens (U+00AD)
    text = text.replace("­", "")

    # Replace non-breaking spaces with regular spaces
    text = text.replace("\xa0", " ")

    # Normalize line endings to \n
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    # Collapse 3+ consecutive newlines → double newline (paragraph break)
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Join hyphenated line breaks: "break-\ning" → "breaking"
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    # Join single line breaks within paragraphs:
    # a line that doesn't end with sentence-ending punctuation → treat as a
    # soft wrap and replace the newline with a space
    text = re.sub(r"(?<![.!?:\"])\n(?=[a-z一-鿿])", " ", text)

    # Collapse repeated spaces and tabs
    text = re.sub(r"[ \t]+", " ", text)

    # Strip leading/trailing whitespace from each line, then the whole text
    lines = [line.strip() for line in text.splitlines()]
    text = "\n".join(line for line in lines if line)

    return text.strip()


# ---------------------------------------------------------------------------
# 2. Page extraction (no 30-page limit)
# ---------------------------------------------------------------------------

def extract_pages_for_rag(pdf_path: str | Path) -> list[dict]:
    """Read a PDF page by page and return readable ``{page, text}`` records.

    Parameters
    ----------
    pdf_path : str or Path
        Path to the PDF file on disk.

    Returns
    -------
    list[dict]
        Each dict has ``page`` (1-based int) and ``text`` (cleaned str).
        Empty pages are skipped.
    """
    pdf_path = Path(pdf_path)
    reader = PdfReader(str(pdf_path))

    pages: list[dict] = []
    for page_number, page_obj in enumerate(reader.pages, start=1):
        raw_text = (page_obj.extract_text() or "").strip()
        cleaned = clean_text(raw_text)
        if cleaned:  # skip empty pages
            pages.append({"page": page_number, "text": cleaned})

    return pages


# ---------------------------------------------------------------------------
# 3. JSON artifact helpers
# ---------------------------------------------------------------------------

def save_json(data, path: str | Path) -> None:
    """Save a Python object to a UTF-8 JSON file, creating parent folders as needed."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path):
    """Read a JSON artifact back into Python."""
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# 4. Notebook inspection helper
# ---------------------------------------------------------------------------

def preview_records(records: list[dict], columns: list[str], rows: int = 5):
    """Show a small notebook table for quick inspection of page/chunk artifacts.

    Parameters
    ----------
    records : list[dict]
        The page or chunk records to preview.
    columns : list[str]
        Which columns to show (e.g. ``["page", "text"]``).
    rows : int
        Number of rows to display (default 5).
    """
    import pandas as pd

    frame = pd.DataFrame(records)
    if frame.empty:
        return frame
    usable_columns = [col for col in columns if col in frame.columns]
    return frame[usable_columns].head(rows)


# ---------------------------------------------------------------------------
# 5. Chunking helpers
# ---------------------------------------------------------------------------

# Internal separator hierarchy for splitting oversized text blocks.
# slice_long_text tries each level in order; finer separators are only used
# when a chunk still exceeds chunk_size with the current separator.
_SEPARATOR_STEPS: list[tuple[str | None, str]] = [
    # (separator_to_join, split_pattern)
    # First: try paragraph boundaries
    ("\n\n", r"\n\n"),
    # Second: try single line breaks within paragraphs
    ("\n", r"\n"),
    # Third: sentence boundaries (period / exclamation / question + whitespace)
    (" ", r"(?<=[.!?])\s+"),
    # Fourth: word boundaries
    (" ", r" "),
    # Last resort: no natural boundary — let caller fall back to character split
]


def _greedy_join(units: list[str], sep: str, chunk_size: int) -> list[str]:
    """Pack units into chunks, each ≤ chunk_size, without splitting units."""
    chunks: list[str] = []
    current = ""
    for unit in units:
        if not unit:
            continue
        candidate = current + (sep if current else "") + unit
        if len(candidate) <= chunk_size:
            current = candidate
        else:
            if current:
                chunks.append(current)
            current = unit
    if current:
        chunks.append(current)
    return chunks


def slice_long_text(text: str, chunk_size: int) -> list[str]:
    """Split a single oversized text block into pieces ≤ *chunk_size*.

    Tries increasingly fine natural boundaries (paragraph → line → sentence →
    word) before falling back to a hard character split.  Always avoids
    splitting in the middle of a word when a separator above character level
    succeeds.
    """
    if len(text) <= chunk_size:
        return [text]

    for join_sep, split_pattern in _SEPARATOR_STEPS:
        units = [u.strip() for u in re.split(split_pattern, text) if u.strip()]
        if not units:
            continue

        # If ANY unit is still too big for this separator, try the next finer one
        if max(len(u) for u in units) > chunk_size:
            continue

        chunks = _greedy_join(units, join_sep, chunk_size)
        if chunks:
            return chunks

    # Fallback: hard character split (should rarely happen — only for text
    # with no spaces, sentences, or paragraphs at all)
    return [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]


def chunk_by_paragraph(pages: list[dict], chunk_size: int) -> list[dict]:
    """Chunk pages by paragraph boundaries.

    Paragraphs within each page are identified by double-newline breaks.  A
    paragraph that fits in *chunk_size* becomes its own chunk; one that
    exceeds it is split further by ``slice_long_text``.
    """
    chunks: list[dict] = []
    chunk_counter = 0

    for page_rec in pages:
        page_num = page_rec["page"]
        paragraphs = [
            p.strip() for p in page_rec["text"].split("\n\n") if p.strip()
        ]

        for para_text in paragraphs:
            pieces = slice_long_text(para_text, chunk_size)
            for piece in pieces:
                chunk_counter += 1
                chunks.append(
                    {
                        "chunk_id": chunk_counter,
                        "page": page_num,
                        "text": piece,
                        "chunk_mode": "paragraph",
                    }
                )

    return chunks


def chunk_by_characters(
    pages: list[dict], chunk_size: int, overlap: int = 0
) -> list[dict]:
    """Chunk pages by fixed-size character windows with optional *overlap*.

    When *overlap* is 0 the mode is ``"character"``; when > 0 it is
    ``"character_overlap"``.  The step size is ``chunk_size - overlap``.
    """
    if overlap >= chunk_size:
        raise ValueError(
            f"overlap ({overlap}) must be smaller than chunk_size ({chunk_size})"
        )

    mode = "character_overlap" if overlap > 0 else "character"
    step = chunk_size - overlap
    if step <= 0:
        step = 1  # safety: should not happen after the check above

    chunks: list[dict] = []
    chunk_counter = 0

    for page_rec in pages:
        page_num = page_rec["page"]
        text = page_rec["text"]
        start = 0
        while start < len(text):
            window = text[start : start + chunk_size]
            if not window.strip():
                start += step
                continue
            chunk_counter += 1
            chunks.append(
                {
                    "chunk_id": chunk_counter,
                    "page": page_num,
                    "text": window,
                    "chunk_mode": mode,
                }
            )
            start += step

    return chunks


# ---------------------------------------------------------------------------
# 6. Embedding pipeline
# ---------------------------------------------------------------------------

# Module-level model cache — one SentenceTransformer per (model_name, device)
_MODEL_CACHE: dict = {}


def model_tag(model_name: str) -> str:
    """Turn a model name into a safe filename suffix.

    ``"sentence-transformers/all-MiniLM-L6-v2"`` → ``"all_minilm_l6_v2"``.
    """
    short = model_name.rsplit("/", 1)[-1]
    safe = re.sub(r"[^a-zA-Z0-9]", "_", short)
    safe = re.sub(r"_+", "_", safe)
    return safe.strip("_").lower()


def resolve_model_source(
    model_name: str, artifact_root: str | Path | None = None
) -> str:
    """Prefer a local cached model folder when it already exists.

    Checks ``{artifact_root}/hf_models/`` for a folder whose name matches the
    short model name or the safe tag.  Falls back to *model_name* (which
    ``sentence-transformers`` will auto-download from HuggingFace).
    """
    short = model_name.rsplit("/", 1)[-1]
    tag = model_tag(model_name)

    if artifact_root is not None:
        root = Path(artifact_root)
        for folder_name in (short, tag):
            candidate = root / "hf_models" / folder_name
            if candidate.exists():
                # Verify it's a real sentence-transformers folder
                required = ["modules.json", "config_sentence_transformers.json"]
                if any((candidate / f).exists() for f in required):
                    return str(candidate)

    return model_name


def get_device() -> str:
    """Return ``"cuda"`` if a GPU is available, otherwise ``"cpu"``."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except ImportError:
        pass
    return "cpu"


def load_model(model_name: str, device: str | None = None):
    """Create or reuse a cached ``SentenceTransformer`` instance.

    The model is loaded only once per *(model_name, device)* pair and reused
    across subsequent calls in the same process.
    """
    if device is None:
        device = get_device()

    cache_key = (model_name, device)
    if cache_key in _MODEL_CACHE:
        return _MODEL_CACHE[cache_key]

    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        model_name,
        device=device,
        model_kwargs={"use_safetensors": False},
    )
    _MODEL_CACHE[cache_key] = model
    return model


def embed_texts(
    texts: list[str],
    model_name: str,
    batch_size: int = 32,
    device: str | None = None,
):
    """Encode a list of texts into normalised ``float32`` vectors.

    Returns
    -------
    np.ndarray
        2-D array of shape ``(len(texts), embedding_dim)``.
    """
    import numpy as np

    model = load_model(model_name, device=device)

    vectors = model.encode(
        texts,
        batch_size=batch_size,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return np.asarray(vectors, dtype=np.float32)


def artifact_paths_for(
    document_id: str,
    chunk_mode: str,
    model_name: str,
    artifact_root: str | Path = "artifacts/rag",
) -> dict:
    """Return file paths for one PDF + config combination.

    Does **not** create files — just returns ``Path`` objects keyed by
    ``raw_pages``, ``chunks``, ``embeddings``, ``manifest``, and ``index``.
    """
    root = Path(artifact_root)
    tag = model_tag(model_name)
    prefix = f"{document_id}_{chunk_mode}"

    return {
        "raw_pages": root / "raw_pages" / f"{document_id}_pages.json",
        "chunks": root / "chunks" / f"{prefix}.json",
        "embeddings": root / "embeddings" / f"{prefix}_{tag}.npy",
        "manifest": root / "embeddings" / f"{prefix}_{tag}.manifest.json",
        "index": root / "indexes" / f"{prefix}_{tag}.faiss",
    }


def ensure_artifacts(
    document_id: str,
    pdf_name: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root: str | Path | None = None,
) -> dict:
    """Build or reuse the pages → chunks → embeddings → manifest bundle.

    On first call the bundle is built from scratch and saved to disk.  On
    subsequent calls with the same configuration the saved artefacts are
    reloaded (cache hit).

    Parameters
    ----------
    document_id : str
        Short label for this document (e.g. ``"pdf1"``).
    pdf_name : str
        Original PDF filename (recorded in the manifest).
    pages : list[dict]
        Page records from ``extract_pages_for_rag``.
    chunk_mode : str
        One of ``"paragraph"``, ``"character"``, ``"character_overlap"``.
    model_name : str
        HuggingFace model id or local folder path.
    chunk_size : int
        Max characters per chunk.
    overlap : int
        Character overlap for sliding-window chunk modes.
    batch_size : int
        Batch size for the embedding model.
    artifact_root : str, Path or None
        Root directory for saved artefacts.  Defaults to
        ``smartlearn-backend/artifacts/rag/``.

    Returns
    -------
    dict
        ``{"pages": …, "chunks": …, "embeddings": np.ndarray, "manifest": …}``
    """
    import numpy as np

    if artifact_root is None:
        artifact_root = (
            Path(__file__).resolve().parent.parent / "artifacts" / "rag"
        )

    paths = artifact_paths_for(document_id, chunk_mode, model_name, artifact_root)

    # --- Cache hit: reload when the config signature still matches ----------
    if paths["manifest"].exists():
        cached = load_json(paths["manifest"])
        config_keys = [
            "document_id", "chunk_mode", "chunk_size", "overlap", "model_name",
        ]
        current = {
            "document_id": document_id,
            "chunk_mode": chunk_mode,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "model_name": model_name,
        }
        if all(cached.get(k) == current[k] for k in config_keys):
            chunks = load_json(paths["chunks"])
            embeddings = np.load(str(paths["embeddings"]))
            return {
                "pages": pages,
                "chunks": chunks,
                "embeddings": embeddings,
                "manifest": cached,
            }

    # --- Build fresh --------------------------------------------------------
    resolved_model = resolve_model_source(model_name, artifact_root)
    device = get_device()

    # Chunk
    chunks = build_chunks(
        pages, chunk_mode=chunk_mode, chunk_size=chunk_size, overlap=overlap
    )
    chunk_texts = [c["text"] for c in chunks]

    # Embed
    embeddings = embed_texts(
        chunk_texts, resolved_model, batch_size=batch_size, device=device
    )

    # Save
    save_json(pages, paths["raw_pages"])
    save_json(chunks, paths["chunks"])
    paths["embeddings"].parent.mkdir(parents=True, exist_ok=True)
    np.save(str(paths["embeddings"]), embeddings)

    manifest = {
        "document_id": document_id,
        "pdf_name": pdf_name,
        "num_pages": len(pages),
        "chunk_mode": chunk_mode,
        "chunk_size": chunk_size,
        "overlap": overlap,
        "model_name": model_name,
        "num_chunks": len(chunks),
        "embedding_dim": int(embeddings.shape[1]),
        "device": device,
        "chunk_path": str(paths["chunks"]),
        "embedding_path": str(paths["embeddings"]),
        "raw_pages_path": str(paths["raw_pages"]),
    }
    save_json(manifest, paths["manifest"])

    return {
        "pages": pages,
        "chunks": chunks,
        "embeddings": embeddings,
        "manifest": manifest,
    }


def build_chunks(
    records: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
) -> list[dict]:
    """Select a chunking strategy and return a uniform chunk schema.

    Parameters
    ----------
    records : list[dict]
        Page records from ``extract_pages_for_rag`` (``[{page, text}]``).
    chunk_mode : str
        One of ``"paragraph"``, ``"character"``, or ``"character_overlap"``.
    chunk_size : int
        Target max characters per chunk (default 700).
    overlap : int
        Overlap in characters for character-based modes (default 120).  Ignored
        by ``"paragraph"`` mode.
    """
    if chunk_mode == "paragraph":
        return chunk_by_paragraph(records, chunk_size)

    if chunk_mode == "character":
        return chunk_by_characters(records, chunk_size, overlap=0)

    if chunk_mode == "character_overlap":
        return chunk_by_characters(records, chunk_size, overlap=overlap)

    raise ValueError(
        f"Unknown chunk_mode '{chunk_mode}'. "
        "Expected one of: 'paragraph', 'character', 'character_overlap'."
    )
