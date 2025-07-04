# server.py
import logging
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import fitz                # PyMuPDF
import openai, os, tempfile, json
import pytesseract
from PIL import Image
from dotenv import load_dotenv

load_dotenv()  # reads .env
openai.api_key = os.getenv("OPENAI_API_KEY")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("notegenie-backend")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # In production, restrict to your domain(s)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- helper: extract text ----------
def extract_text(file: UploadFile) -> str:
    ext = file.filename.split('.')[-1].lower()
    with tempfile.NamedTemporaryFile(delete=False, suffix='.' + ext) as tmp:
        tmp.write(file.file.read())
        tmp_path = tmp.name

    try:
        if ext == "pdf":
            doc = fitz.open(tmp_path)
            text = "\n".join(page.get_text() for page in doc)
        elif ext in {"png", "jpg", "jpeg"}:
            image = Image.open(tmp_path)
            text = pytesseract.image_to_string(image)
        else:  # txt / md / docx handled as plain text
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    finally:
        os.unlink(tmp_path)

    return text.strip()

# ---------- helper: call OpenAI ----------
def call_openai(prompt: str, system_msg: str = "You are a helpful study assistant.") -> str:
    logger.info("Calling OpenAI with prompt length: %d", len(prompt))
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
        temperature=0.5,
    )
    result = response.choices[0].message.content.strip()
    logger.info("OpenAI response length: %d", len(result))
    return result

# ---------- main endpoint ----------
@app.post("/api/process")
async def process(file: UploadFile, mode: str = Form(...)):
    try:
        text = extract_text(file)[:15_000]  # safeguard token length
        if not text:
            logger.warning("No readable text found in uploaded file")
            return JSONResponse({"error": "No readable text found"}, status_code=400)

        if mode == "summary":
            prompt = f"Summarize the following notes in concise bullet points:\n\n{text}"
            summary = call_openai(prompt)
            return {"summary": summary}

        elif mode == "flashcards":
            prompt = (
                "Generate 5 Q&A flashcards from the notes below. "
                'Return JSON: [{"question":"...","answer":"..."}, ...]\n\n' + text
            )
            response_text = call_openai(prompt)
            try:
                flashcards = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error("JSON parse error in flashcards: %s", e)
                return JSONResponse({"error": "Failed to parse flashcards JSON"}, status_code=500)
            return {"flashcards": flashcards}

        elif mode == "quiz":
            prompt = (
                "Create 5 multiple-choice questions (A-D options) from these notes. "
                'Return JSON: [{"question":"...","options":["A","B","C","D"],"answer":"B"}, ...]\n\n' + text
            )
            response_text = call_openai(prompt)
            try:
                questions = json.loads(response_text)
            except json.JSONDecodeError as e:
                logger.error("JSON parse error in quiz questions: %s", e)
                return JSONResponse({"error": "Failed to parse quiz questions JSON"}, status_code=500)
            return {"questions": questions}

        else:
            logger.warning("Invalid mode requested: %s", mode)
            return JSONResponse({"error": "Invalid mode"}, status_code=400)

    except Exception as e:
        logger.exception("Unexpected error occurred:")
        return JSONResponse({"error": str(e)}, status_code=500)
