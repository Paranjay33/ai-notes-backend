# AI Notes Summarizer Backend (FastAPI)

This backend processes uploaded notes (PDF, text, or image), extracts text, and generates:
- Summaries
- Flashcards
- Quiz Questions

Built with FastAPI + OpenAI API.

## Endpoint

POST `/api/process`
- `file`: PDF / image / text file
- `mode`: 'summary' | 'flashcards' | 'quiz'
