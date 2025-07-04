# server.py
from fastapi import FastAPI, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import fitz                # PyMuPDF
import openai, os, tempfile, json
import pytesseract
from PIL import Image
from dotenv import load_dotenv

load_dotenv()                               # reads .env
openai.api_key = os.getenv("OPENAI_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],     # in prod you can restrict to your domain
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
        else:                   # txt / md / docx handled as plain text
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
    finally:
        os.unlink(tmp_path)

    return text.strip()

# ---------- helper: call OpenAI ----------
def call_openai(prompt: str, system_msg: str = "You are a helpful study assistant.") -> str:
    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user",   "content": prompt},
        ],
        temperature=0.5,
    )
    return response.choices[0].message.content.strip()

# ---------- main endpoint ----------
@app.post("/api/process")
async def process(file: UploadFile, mode: str = Form(...)):
    try:
        text = extract_text(file)[:15_000]          # safeguard token length
        if not text:
            return JSONResponse({"error": "No readable text found"}, status_code=400)

        if mode == "summary":
            prompt = f"""Summarize the following notes in concise bullet points:\n\n{text}"""
            return {"summary": call_openai(prompt)}

        elif mode == "flashcards":
            prompt = (
                "Generate 5 Q&A flashcards from the notes below. "
                'Return JSON: [{"question":"...","answer":"..."}, ...]\n\n' + text
            )
            return {"flashcards": json.loads(call_openai(prompt))}

        elif mode == "quiz":
            prompt = (
                "Create 5 multiple‑choice questions (A‑D options) from these notes. "
                'Return JSON: [{"question":"...","options":["A","B","C","D"],"answer":"B"}, ...]\n\n' + text
            )
            return {"questions": json.loads(call_openai(prompt))}

        else:
            return JSONResponse({"error": "Invalid mode"}, status_code=400)

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)
