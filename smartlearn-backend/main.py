import os
import re

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from pathlib import Path

from services.llm import answer_from_pages
from services.pdf import extract_pages
from services import rag

app = FastAPI(title="SmartLearn Lite API")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
allowed_origins = [
    origin.strip() for origin in ALLOWED_ORIGINS.split(",") if origin.strip()
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=False,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

documents: dict[str, dict] = {}


class ChatRequest(BaseModel):
    chat_id: str = "day2-demo"
    message: str = Field(..., min_length=2, max_length=2000)


@app.get("/")
def root():
    return {"message": "SmartLearn Lite API is running"}


@app.get("/health")
def health():
    return {"ok": True}


@app.post("/upload")
async def upload(
    chat_id: str = Query(..., description="Chat session ID"),
    file: UploadFile = File(..., description="PDF file to upload"),
):
    if not file.content_type or file.content_type != "application/pdf":
        raise HTTPException(status_code=400, detail="Only PDF files are accepted")

    pdf_bytes = await file.read()

    if not pdf_bytes:
        raise HTTPException(status_code=400, detail="Empty file is not accepted")

    try:
        document = rag.prepare_rag_chat_record(
            chat_id=chat_id,
            filename=file.filename or "uploaded.pdf",
            pdf_bytes=pdf_bytes,
        )
    except (ValueError, Exception) as e:
        raise HTTPException(status_code=400, detail=str(e))

    if document.get("characters", 0) == 0:
        raise HTTPException(status_code=422, detail="PDF contains no readable text — OCR is not supported")

    documents[chat_id] = document

    return rag.build_upload_response(document)


@app.get("/documents/{chat_id}/file")
def get_document_file(chat_id: str):
    """Serve the uploaded PDF file for a given chat session."""
    record = documents.get(chat_id)
    if record is None:
        raise HTTPException(status_code=404, detail=f"No document found for chat_id '{chat_id}'")

    file_path = record.get("saved_pdf_path")
    if not file_path or not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="Saved PDF file is missing")

    return FileResponse(str(file_path), media_type="application/pdf")


@app.post("/chat")
def chat(body: ChatRequest):
    document = documents.get(body.chat_id)
    if document is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for chat_id '{body.chat_id}'. Upload a PDF first via POST /upload?chat_id={body.chat_id}",
        )

    try:
        result = rag.answer_chat_turn(document, body.message)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream AI service failed: {e}")

    return {
        "answer": result["answer"],
        "citations": result["citations"],
        "sources": result["sources"],
    }
