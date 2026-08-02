import json
import os

import requests
from dotenv import load_dotenv

load_dotenv()

SYSTEM_PROMPT = (
    "You answer messages only from the supplied PDF text. "
    "Cite factual claims with [Page X]. "
    "If the answer is not in the PDF, say that the document does not provide enough information. "
    "Never invent a page number."
)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def answer_from_pages(pages: list[dict], message: str) -> str:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY is not configured")

    document_text = "\n\n".join(
        f"### [Page {page['page']}]\n{page['text']}"
        for page in pages
        if page["text"]
    )

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": os.getenv("OPENROUTER_MODEL", "openrouter/free"),
        "temperature": 0.0,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"PDF text:\n{document_text}\n\nmessage: {message}"},
        ],
    }

    resp = requests.post(OPENROUTER_URL, headers=headers, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data["choices"][0]["message"]["content"] or ""
