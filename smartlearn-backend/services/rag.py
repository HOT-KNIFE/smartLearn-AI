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


# ---------------------------------------------------------------------------
# 7. FAISS index helpers
# ---------------------------------------------------------------------------


def relative_path_str(path: str | Path, base: str | Path) -> str:
    """Return *path* as a string relative to *base* for display.

    When *path* is not under *base*, return the absolute path as a fallback.
    """
    try:
        return str(Path(path).resolve().relative_to(Path(base).resolve()))
    except ValueError:
        return str(Path(path).resolve())


def build_faiss_index(embeddings: "np.ndarray") -> "faiss.Index":
    """Create a FAISS inner-product index from **normalised** embedding vectors.

    Because ``embed_texts`` already L2-normalises every vector, inner product
    is equivalent to cosine similarity.  ``IndexFlatIP`` is exact (not
    approximate) and is the right choice at the hundreds-of-chunks scale.
    """
    import faiss
    import numpy as np

    embeddings = np.asarray(embeddings, dtype=np.float32)
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    return index


def save_faiss_index(index: "faiss.Index", index_path: str | Path) -> None:
    """Write a FAISS index to a binary ``.faiss`` file on disk."""
    import faiss

    path = Path(index_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(path))


def load_faiss_index(index_path: str | Path) -> "faiss.Index":
    """Read a FAISS index from a saved ``.faiss`` file."""
    import faiss

    return faiss.read_index(str(index_path))


def ensure_index(
    document_id: str,
    pdf_name: str,
    pages: list[dict] | None = None,
    pdf_path: str | Path | None = None,
    chunk_mode: str = "character_overlap",
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    chunk_size: int = 700,
    overlap: int = 120,
    batch_size: int = 32,
    artifact_root: str | Path | None = None,
) -> dict:
    """Build or reuse the full chunks → embeddings → FAISS index bundle.

    Reuses ``ensure_artifacts`` for the chunk + embedding cache layer.  When
    the ``.faiss`` index file is missing it is built from the (already
    normalised) embeddings and saved alongside an index metadata file.

    Parameters
    ----------
    document_id : str
        Short label (e.g. ``"pdf1"``).
    pdf_name : str
        Original PDF filename (recorded in manifests).
    pages : list[dict] or None
        Page records from ``extract_pages_for_rag``.  If ``None``, extracted
        from *pdf_path*.
    pdf_path : str, Path or None
        Path to the source PDF.  Only used when *pages* is ``None``.
    chunk_mode / model_name / chunk_size / overlap / batch_size / artifact_root :
        Forwarded to ``ensure_artifacts``.

    Returns
    -------
    dict
        ``{"pages", "chunks", "embeddings", "manifest", "index_path",
        "chunk_paths", "model_source"}``
    """
    import numpy as np

    # --- Resolve pages -------------------------------------------------------
    if pages is None:
        if pdf_path is None:
            raise ValueError("Either pages or pdf_path must be provided.")
        pages = extract_pages_for_rag(pdf_path)

    # --- Chunks + embeddings (cached) ----------------------------------------
    bundle = ensure_artifacts(
        document_id=document_id,
        pdf_name=pdf_name,
        pages=pages,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )

    # --- Resolve paths -------------------------------------------------------
    paths = artifact_paths_for(document_id, chunk_mode, model_name,
                               artifact_root or (Path(__file__).resolve().parent.parent / "artifacts" / "rag"))

    index_path = paths["index"]
    index_meta_path = index_path.with_suffix(".index_meta.json")

    # --- Build FAISS index only when missing ---------------------------------
    if not index_path.exists():
        index = build_faiss_index(bundle["embeddings"])
        save_faiss_index(index, index_path)

        index_meta = {
            "document_id": document_id,
            "pdf_name": pdf_name,
            "num_vectors": int(bundle["embeddings"].shape[0]),
            "embedding_dim": int(bundle["embeddings"].shape[1]),
            "chunk_mode": chunk_mode,
            "chunk_size": chunk_size,
            "overlap": overlap,
            "model_name": model_name,
        }
        save_json(index_meta, index_meta_path)

    # --- Load index for in-memory use ----------------------------------------
    index = load_faiss_index(index_path)

    return {
        "pages": bundle["pages"],
        "chunks": bundle["chunks"],
        "embeddings": bundle["embeddings"],
        "manifest": bundle["manifest"],
        "index": index,
        "index_path": str(index_path),
        "chunk_paths": paths,
        "model_source": resolve_model_source(model_name, artifact_root),
    }


def prepare_rag_document(
    document_id: str,
    filename: str,
    pages: list[dict],
    chunk_mode: str = "character_overlap",
    chunk_size: int = 700,
    overlap: int = 120,
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    batch_size: int = 32,
    artifact_root: str | Path | None = None,
) -> dict:
    """Prepare one server-style document record with chunks, embeddings, and index.

    This is the Lab B document shape that later fits into the Day 2
    ``documents[chat_id]`` store.  It calls ``ensure_index`` internally and
    wraps the result with the fields the app and notebook expect.

    Parameters
    ----------
    document_id : str
        Short label (e.g. ``"pdf1"``).
    filename : str
        Original PDF filename (e.g. ``"pdf1.pdf"``).
    pages : list[dict]
        Page records from ``extract_pages_for_rag``.
    chunk_mode / chunk_size / overlap / model_name / batch_size / artifact_root :
        Forwarded to ``ensure_index``.

    Returns
    -------
    dict
        A document record with ``document_id``, ``filename``, ``pages``,
        ``chunks``, ``chunk_size``, ``embedding_dim``, ``model_name``,
        ``model_source``, ``chunk_mode``, ``artifacts`` (paths dict), and
        ``history`` (empty list).
    """
    bundle = ensure_index(
        document_id=document_id,
        pdf_name=filename,
        pages=pages,
        chunk_mode=chunk_mode,
        model_name=model_name,
        chunk_size=chunk_size,
        overlap=overlap,
        batch_size=batch_size,
        artifact_root=artifact_root,
    )

    paths = bundle["chunk_paths"]

    return {
        "document_id": document_id,
        "filename": filename,
        "pages": bundle["pages"],
        "chunks": bundle["chunks"],
        "chunk_size": len(bundle["chunks"]),
        "embedding_dim": bundle["manifest"]["embedding_dim"],
        "model_name": model_name,
        "model_source": bundle["model_source"],
        "chunk_mode": chunk_mode,
        "artifacts": {
            "index": bundle["index_path"],
            "chunks": str(paths["chunks"]),
            "embeddings": str(paths["embeddings"]),
            "manifest": str(paths["manifest"]),
        },
        "history": [],
    }


# ---------------------------------------------------------------------------
# 8. Retrieval helpers
# ---------------------------------------------------------------------------

# Common English stopwords filtered in keyword_set.
_STOPWORDS: set[str] = {
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would", "could",
    "should", "may", "might", "can", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "this", "that", "these", "those",
    "it", "its", "and", "or", "but", "not", "no", "if", "as", "so",
    "we", "he", "she", "they", "you", "i", "me", "my", "your", "his",
    "her", "our", "what", "which", "who", "whom", "when", "where", "how",
    "all", "also", "into", "than", "then", "about", "up", "out", "just",
    "one", "two", "each", "some", "any", "very", "only", "other", "more",
}


def keyword_set(text: str) -> set[str]:
    """Extract lightweight lexical tokens for simple reranking.

    English words (2+ alphanumeric chars) are extracted after lowercasing.
    Common stopwords are removed so only content-bearing tokens remain.

    Parameters
    ----------
    text : str
        A question or chunk text.

    Returns
    -------
    set[str]
        Lowercased content tokens ready for Jaccard-style overlap scoring.
    """
    if not text:
        return set()

    text = text.lower()

    # English words (2+ alphanumeric chars)
    tokens: list[str] = re.findall(r"[a-z0-9]{2,}", text)

    return {t for t in tokens if t not in _STOPWORDS}


def split_sentences(text: str) -> list[str]:
    """Split a chunk text into candidate answer sentences.

    Sentence boundaries are detected by end-of-sentence punctuation
    (``. ! ?``) followed by whitespace.

    Parameters
    ----------
    text : str
        Retrieved chunk text.

    Returns
    -------
    list[str]
        Non-empty sentence strings with leading / trailing whitespace stripped.
    """
    if not text:
        return []

    # Split on sentence-ending punctuation + whitespace, keeping the
    # delimiter on the preceding sentence.
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def search_bundle(
    question: str,
    bundle: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    batch_size: int = 1,
    history: list[dict] | None = None,
) -> list[dict]:
    """Search an in-memory index bundle for chunks relevant to *question*.

    Embeds the question with the same model, performs FAISS inner-product
    search over *candidate_pool* neighbours, applies a small lexical-rerank
    bonus for keyword overlap, and returns the final *top_k* hits.

    Parameters
    ----------
    question : str
        Natural-language question.
    bundle : dict
        In-memory bundle from ``ensure_index``.  Must contain ``index``
        (faiss.Index), ``chunks`` (list[dict]), and ``manifest`` (dict with at
        least ``model_name``).
    top_k : int
        Number of hits to return (default 3).
    candidate_pool : int
        How many neighbours to fetch from FAISS before reranking (default 60).
    batch_size : int
        Batch size passed to the embedding model (default 1).
    history : list[dict] or None
        Optional conversation history; accepted for API compatibility and
        ignored by the current implementation.

    Returns
    -------
    list[dict]
        Up to *top_k* hits, each with ``page``, ``chunk_id``, ``text``, and
        ``score`` keys.
    """
    import numpy as np

    # --- Resolve model from the bundle ----------------------------------------
    manifest = bundle["manifest"]
    model_name = bundle.get("model_source", manifest.get("model_name"))

    # --- Embed the question ---------------------------------------------------
    q_vec = embed_texts(
        [question], model_name, batch_size=batch_size,
    )
    # embed_texts returns (1, dim); ensure float32 contiguous for FAISS
    q_vec = np.asarray(q_vec, dtype=np.float32)

    # --- Semantic retrieval via FAISS -----------------------------------------
    index = bundle["index"]
    k = min(candidate_pool, index.ntotal)
    scores, indices = index.search(q_vec, k)

    chunks = bundle["chunks"]

    hits: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        hits.append({
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": float(score),
        })

    # --- Lightweight lexical rerank -------------------------------------------
    # A small bonus (max +0.1) is added based on keyword overlap between the
    # question and each hit's text.  This nudges chunks that share more
    # content words higher without overriding the semantic score.
    if len(hits) > top_k:
        q_keywords = keyword_set(question)
        if q_keywords:
            for hit in hits:
                hit_kw = keyword_set(hit["text"])
                if hit_kw:
                    overlap_ratio = len(q_keywords & hit_kw) / len(q_keywords)
                    hit["score"] = hit["score"] + 0.1 * overlap_ratio

        # Re-sort by adjusted score
        hits.sort(key=lambda h: h["score"], reverse=True)

    return hits[:top_k]


def search_document(
    question: str,
    document: dict,
    top_k: int = 3,
    candidate_pool: int = 60,
    history: list[dict] | None = None,
) -> list[dict]:
    """Search a prepared document record for chunks relevant to *question*.

    Loads the FAISS index and chunk metadata from disk (as saved by
    ``prepare_rag_document``), embeds the question, and returns top-k hits.

    Parameters
    ----------
    question : str
        Natural-language question.
    document : dict
        Document record from ``prepare_rag_document``.  Must contain
        ``artifacts`` (with ``index`` and ``chunks`` paths), ``model_name``,
        and optionally ``model_source``.
    top_k : int
        Number of hits to return (default 3).
    candidate_pool : int
        How many neighbours to fetch from FAISS before reranking (default 60).
    history : list[dict] or None
        Optional conversation history; accepted for API compatibility and
        ignored by the current implementation.

    Returns
    -------
    list[dict]
        Up to *top_k* hits, each with ``page``, ``chunk_id``, ``text``, and
        ``score`` keys.
    """
    import numpy as np

    # --- Load FAISS index and chunks from disk --------------------------------
    index_path = document["artifacts"]["index"]
    index = load_faiss_index(index_path)

    chunks_path = document["artifacts"]["chunks"]
    chunks = load_json(chunks_path)

    # --- Resolve model --------------------------------------------------------
    model_name = document.get("model_source", document["model_name"])

    # --- Embed the question ---------------------------------------------------
    q_vec = embed_texts([question], model_name, batch_size=1)
    q_vec = np.asarray(q_vec, dtype=np.float32)

    # --- Semantic retrieval ---------------------------------------------------
    k = min(candidate_pool, index.ntotal)
    scores, indices = index.search(q_vec, k)

    hits: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(chunks):
            continue
        chunk = chunks[idx]
        hits.append({
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "score": float(score),
        })

    # --- Lightweight lexical rerank -------------------------------------------
    if len(hits) > top_k:
        q_keywords = keyword_set(question)
        if q_keywords:
            for hit in hits:
                hit_kw = keyword_set(hit["text"])
                if hit_kw:
                    overlap_ratio = len(q_keywords & hit_kw) / len(q_keywords)
                    hit["score"] = hit["score"] + 0.1 * overlap_ratio

        hits.sort(key=lambda h: h["score"], reverse=True)

    return hits[:top_k]


def best_sentence_answer(question: str, hits: list[dict]) -> str:
    """Pick the single best answer sentence from a set of retrieval hits.

    Each hit's text is split into sentences.  The sentence with the highest
    keyword-overlap (Jaccard) against *question* wins.  When the best sentence
    belongs to a hit with a known page, a ``(p.N)`` tag is prepended.

    Parameters
    ----------
    question : str
        Natural-language question.
    hits : list[dict]
        Retrieval hits, each with at least ``text`` and optionally ``page``.

    Returns
    -------
    str
        The best sentence, optionally prefixed with a page tag
        (e.g. ``"(p.3) The answer sentence."``).  Returns an empty string when
        no suitable sentence is found.
    """
    if not question or not hits:
        return ""

    q_keywords = keyword_set(question)
    if not q_keywords:
        return ""

    best_sentence = ""
    best_score = -1.0
    best_page: int | None = None

    for hit in hits:
        sentences = split_sentences(hit.get("text", ""))
        page = hit.get("page")
        for sent in sentences:
            sent_kw = keyword_set(sent)
            if not sent_kw:
                continue
            # Jaccard similarity: intersection / union
            intersection = q_keywords & sent_kw
            union = q_keywords | sent_kw
            jaccard = len(intersection) / len(union)
            if jaccard > best_score:
                best_score = jaccard
                best_sentence = sent
                best_page = page

    if not best_sentence:
        return ""

    if best_page is not None:
        return f"(p.{best_page}) {best_sentence}"
    return best_sentence


# ---------------------------------------------------------------------------
# 9. Project-facing helpers (Lab B §2.5)
# ---------------------------------------------------------------------------


def extract_citations(answer: str, hits: list[dict] | None = None) -> list[int]:
    """Extract numeric PDF page citations from an answer and optional hits.

    Scans the answer for page-tag patterns (``(p.N)``, ``[Page N]``) and
    also collects unique page numbers from *hits* when provided.  Results are
    deduplicated and sorted.

    Parameters
    ----------
    answer : str
        An answer string that may contain page citations.
    hits : list[dict] or None
        Optional retrieval hits.  When provided their ``page`` fields are
        merged into the output.

    Returns
    -------
    list[int]
        Unique, sorted page numbers cited in the answer or present in *hits*.
    """
    pages: set[int] = set()

    # --- Scan answer text for page patterns ---------------------------------
    if answer:
        # "(p.3)", "(p. 3)"
        for m in re.finditer(r"\(p\.\s*(\d+)\)", answer, re.IGNORECASE):
            pages.add(int(m.group(1)))
        # "[Page 3]", "[Page3]"
        for m in re.finditer(r"\[Page\s*(\d+)\]", answer, re.IGNORECASE):
            pages.add(int(m.group(1)))

    # --- Merge page numbers from hits ---------------------------------------
    if hits:
        for hit in hits:
            page = hit.get("page")
            if page is not None:
                pages.add(int(page))

    return sorted(pages)


def build_sources(hits: list[dict]) -> list[dict]:
    """Build frontend-friendly source objects from retrieval hits.

    Each source carries enough information for the UI to render a clickable
    page link and a short preview of the evidence text.

    Parameters
    ----------
    hits : list[dict]
        Retrieval hits, each with ``page``, ``chunk_id``, ``text``, and
        ``score`` keys.

    Returns
    -------
    list[dict]
        Source objects with ``page``, ``chunk_id``, ``score``, and ``preview``
        (first 200 characters of the chunk text).
    """
    sources: list[dict] = []
    for hit in hits:
        text = hit.get("text", "")
        preview = text[:200] if len(text) > 200 else text
        sources.append({
            "page": hit.get("page"),
            "chunk_id": hit.get("chunk_id"),
            "score": hit.get("score"),
            "preview": preview,
        })
    return sources


def answer_document(
    document: dict,
    question: str,
    top_k: int = 3,
    candidate_pool: int = 60,
    answer_model: str = "openrouter/free",
) -> dict:
    """Answer one question against a prepared document with retrieval + LLM.

    Retrieves top-k chunks from the saved FAISS index and builds an answer:
    - When ``OPENROUTER_API_KEY`` is set in the environment the retrieved
      chunks are sent to the LLM along with a system prompt that requires
      page-backed citations.
    - When the API key is missing the function falls back to
      ``best_sentence_answer``, producing a local extracted answer without
      any external API call.

    Parameters
    ----------
    document : dict
        Document record from ``prepare_rag_document``.
    question : str
        Natural-language question.
    top_k : int
        Number of chunks to retrieve (default 3).
    candidate_pool : int
        FAISS neighbour pool size before reranking (default 60).
    answer_model : str
        OpenRouter model id used when an API key is available (default
        ``"openrouter/free"``).

    Returns
    -------
    dict
        ``{"answer": str, "citations": list[int], "sources": list[dict]}``
    """
    import os

    # --- Retrieve -----------------------------------------------------------
    hits = search_document(
        question, document,
        top_k=top_k,
        candidate_pool=candidate_pool,
    )

    # --- Answer -------------------------------------------------------------
    api_key = os.getenv("OPENROUTER_API_KEY")
    if api_key:
        answer = _llm_answer_from_hits(question, hits, answer_model, api_key)
    else:
        answer = best_sentence_answer(question, hits)

    # --- Citations & sources ------------------------------------------------
    citations = extract_citations(answer, hits)
    sources = build_sources(hits)

    return {
        "answer": answer,
        "citations": citations,
        "sources": sources,
    }


def _llm_answer_from_hits(
    question: str, hits: list[dict], model: str, api_key: str
) -> str:
    """Call OpenRouter with retrieved chunks as context.

    Internal helper — not part of the public API.
    """
    import requests

    # Build a compact context from the retrieved chunks
    context_blocks: list[str] = []
    for hit in hits:
        page = hit.get("page", "?")
        text = hit.get("text", "")
        context_blocks.append(f"### [Page {page}]\n{text}")

    context = "\n\n".join(context_blocks)

    system_prompt = (
        "You answer messages only from the supplied PDF text. "
        "Cite factual claims with [Page X]. "
        "If the answer is not in the PDF, say that the document does not "
        "provide enough information. "
        "Never invent a page number."
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": model,
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"PDF text:\n{context}\n\nQuestion: {question}"},
        ],
    }

    url = "https://openrouter.ai/api/v1/chat/completions"
    resp = requests.post(url, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"] or ""


def append_history(
    document: dict, question: str, result: dict
) -> list[dict]:
    """Record a question-answer pair in the document's in-memory history.

    The entry is appended to ``document["history"]`` in place and the updated
    list is also returned for convenience.

    Parameters
    ----------
    document : dict
        Document record from ``prepare_rag_document``.  Mutated in place.
    question : str
        The user question.
    result : dict
        Answer result from ``answer_document`` (must contain at least
        ``"answer"``).

    Returns
    -------
    list[dict]
        The updated history list.  Each entry has ``question``, ``answer``,
        and ``citations`` keys.
    """
    entry = {
        "question": question,
        "answer": result.get("answer", ""),
        "citations": result.get("citations", []),
    }
    document["history"].append(entry)
    return document["history"]


# ---------------------------------------------------------------------------
# 10. Evaluation helpers (Lab B §2.9)
# ---------------------------------------------------------------------------


def normalize_for_match(text: str) -> str:
    """Normalize text for simple string-based scoring.

    Lowercases, strips whitespace, collapses repeated spaces, and removes
    common punctuation differences so that fuzzy string comparisons have a
    higher chance of matching.

    Parameters
    ----------
    text : str
        Raw extracted text or a gold answer string.

    Returns
    -------
    str
        Normalized, comparison-ready text.
    """
    if not text:
        return ""

    text = text.lower().strip()
    # Collapse repeated whitespace
    text = re.sub(r"\s+", " ", text)
    # Normalise common punctuation variants
    text = text.replace("–", "-").replace("—", "-")
    text = text.replace(""", '"').replace(""", '"')
    text = text.replace("'", "'").replace("'", "'")

    return text


def contains_any_answer(text: str, answers: list[str]) -> bool:
    """Check whether *text* contains at least one acceptable answer.

    Both *text* and each gold answer are normalised with
    ``normalize_for_match`` before the substring check.

    Parameters
    ----------
    text : str
        A text block (e.g. an extracted answer or chunk content).
    answers : list[str]
        Acceptable answer strings.

    Returns
    -------
    bool
        ``True`` when at least one normalised answer appears inside the
        normalised text.
    """
    norm_text = normalize_for_match(text)
    for ans in answers:
        norm_ans = normalize_for_match(ans)
        if norm_ans in norm_text:
            return True
    return False


def evaluate_questions(
    eval_set: list[dict],
    documents_by_name: dict[str, dict],
    top_k: int = 3,
    candidate_pool: int = 60,
) -> "pd.DataFrame":
    """Run a small evaluation and return a table of results.

    For each question the function retrieves chunks, extracts a local answer,
    and scores two binary metrics:

    - **retrieval_hit**: at least one gold answer appears in the retrieved
      chunk text (evidence was found).
    - **answer_hit**: the extracted answer contains at least one gold answer
      (the answer is correct).

    Parameters
    ----------
    eval_set : list[dict]
        Each entry must have ``"pdf_name"``, ``"question"``, and
        ``"answers"`` (list of acceptable gold strings).
    documents_by_name : dict[str, dict]
        Mapping from ``pdf_name`` to prepared document record
        (from ``prepare_rag_document``).
    top_k : int
        Number of chunks to retrieve per question (default 3).
    candidate_pool : int
        FAISS neighbour pool size (default 60).

    Returns
    -------
    pandas.DataFrame
        One row per question with columns: ``pdf_name``, ``question``,
        ``gold_answers``, ``local_answer``, ``pages``, ``retrieval_hit``,
        ``answer_hit``.
    """
    import pandas as pd

    rows: list[dict] = []

    for item in eval_set:
        pdf_name = item["pdf_name"]
        question = item["question"]
        gold_answers = item["answers"]

        document = documents_by_name.get(pdf_name)
        if document is None:
            rows.append({
                "pdf_name": pdf_name,
                "question": question,
                "gold_answers": gold_answers,
                "local_answer": "(document not found)",
                "pages": [],
                "retrieval_hit": False,
                "answer_hit": False,
            })
            continue

        # Retrieve
        hits = search_document(
            question, document, top_k=top_k, candidate_pool=candidate_pool,
        )

        # Local answer
        answer = best_sentence_answer(question, hits)

        # Check retrieval: do the retrieved chunks contain any gold answer?
        all_chunk_text = " ".join(h.get("text", "") for h in hits)
        retrieval_hit = contains_any_answer(all_chunk_text, gold_answers)

        # Check answer: does the generated answer contain any gold answer?
        answer_hit = contains_any_answer(answer, gold_answers)

        rows.append({
            "pdf_name": pdf_name,
            "question": question,
            "gold_answers": gold_answers,
            "local_answer": answer,
            "pages": sorted({h["page"] for h in hits}),
            "retrieval_hit": retrieval_hit,
            "answer_hit": answer_hit,
        })

    return pd.DataFrame(rows)
