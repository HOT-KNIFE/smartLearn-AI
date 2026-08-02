import os
import re

from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .services.llm import answer_from_pages
from .services.pdf import extract_pages

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

documents: dict[str, list[dict]] = {}


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
        pages = extract_pages(pdf_bytes)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    total_chars = sum(len(p["text"]) for p in pages)
    if total_chars == 0:
        raise HTTPException(status_code=422, detail="PDF contains no readable text — OCR is not supported")

    documents[chat_id] = pages

    return {
        "status": "ok",
        "filename": file.filename,
        "pages": len(pages),
        "characters": total_chars,
    }


@app.post("/chat")
def chat(body: ChatRequest):
    pages = documents.get(body.chat_id)
    if pages is None:
        raise HTTPException(
            status_code=404,
            detail=f"No document found for chat_id '{body.chat_id}'. Upload a PDF first via POST /upload?chat_id={body.chat_id}",
        )

    try:
        answer = answer_from_pages(pages, body.message)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Upstream AI service failed: {e}")

    all_page_numbers = {p["page"] for p in pages}
    cited = {int(m) for m in re.findall(r"\[Page (\d+)\]", answer)}
    citations = sorted(c for c in cited if c in all_page_numbers)

    return {"answer": answer, "citations": citations}
