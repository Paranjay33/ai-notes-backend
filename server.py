# server.py
import json
import logging
import os
import tempfile
from typing import Dict, Any

import fitz                    # PyMuPDF
import pytesseract
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from openai import OpenAI      # new SDK (≥ 1.0)

# --------------------------------------------------------------------------- #
#  Environment & Client setup
# --------------------------------------------------------------------------- #
load_dotenv()
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not set in the environment (.env)")

client = OpenAI(api_key=OPENAI_API_KEY)

# --------------------------------------------------------------------------- #
#  FastAPI application & CORS
# --------------------------------------------------------------------------- #
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],        # tighten this in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --------------------------------------------------------------------------- #
#  Logging
# --------------------------------------------------------------------------- #
logging.basicConfig(level=logging.INFO, format="%(levelname)s:%(name)s: %(message)s")
logger = logging.getLogger("notegenie-backend")

# --------------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------------- #
def extract_text(upload: UploadFile) -> str:
    """Extract text from PDF, image (OCR) or plain‑text file."""
    suffix = upload.filename.rsplit(".", 1)[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{suffix}") as tmp:
        tmp.write(upload.file.read())
        path = tmp.name

    try:
        if suffix == "pdf":
            doc = fitz.open(path)
            text = "\n".join(page.get_text() for page in doc)
        elif suffix in {"png", "jpg", "jpeg"}:
            text = pytesseract.image_to_string(Image.open(path))
        else:
            with open(path, "r", encoding="utf‑8", errors="ignore") as fh:
                text = fh.read()
    finally:
        os.unlink(path)

    return text.strip()


def call_openai(prompt: str, system: str = "You are a helpful study assistant.") -> str:
    """Single call to OpenAI Chat Completion (new SDK syntax)."""
    logger.info("Calling OpenAI | prompt length %d chars", len(prompt))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",           # change if you have a different entitlement
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    content = resp.choices[0].message.content.strip()
    logger.info("OpenAI response length %d chars", len(content))
    return content


def safe_json_loads(raw: str, kind: str) -> Dict[str, Any]:
    """Parse JSON returned by the model, raising a clear 500 on failure."""
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.error("JSON %s parsing failed: %s", kind, exc)
        raise ValueError(f"Failed to parse {kind} JSON from OpenAI response.") from exc


# --------------------------------------------------------------------------- #
#  Main endpoint
# --------------------------------------------------------------------------- #
@app.post("/api/process")
async def process(file: UploadFile, mode: str = Form(...)):
    try:
        text = extract_text(file)[:15_000]  # length guard for tokens
        if not text:
            return JSONResponse({"error": "No readable text found in the file."}, status_code=400)

        if mode == "summary":
            prompt = f"Summarise the following notes in concise bullet points:\n\n{text}"
            summary = call_openai(prompt)
            return {"summary": summary}

        if mode == "flashcards":
            prompt = (
                "Generate exactly five Q‑and‑A flashcards from the notes below. "
                'Return *only* valid JSON in the form '
                '[{"question":"...","answer":"..."}, …]\n\n' + text
            )
            flashcards = safe_json_loads(call_openai(prompt), "flashcards")
            return {"flashcards": flashcards}

        if mode == "quiz":
            prompt = (
                "Create exactly five multiple‑choice questions (options A‑D) from these notes. "
                'Return *only* valid JSON in the form '
                '[{"question":"...","options":["A","B","C","D"],"answer":"B"}, …]\n\n' + text
            )
            questions = safe_json_loads(call_openai(prompt), "quiz‑questions")
            return {"questions": questions}

        return JSONResponse({"error": "Invalid mode selected."}, status_code=400)

    except ValueError as ve:         # JSON parsing or other validation
        return JSONResponse({"error": str(ve)}, status_code=500)
    except Exception as exc:         # catch‑all
        logger.exception("Unexpected error")
        return JSONResponse({"error": str(exc)}, status_code=500)
