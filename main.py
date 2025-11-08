from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel 
from summarizer import generate_summary


app = FastAPI(title="AI Text Summarizer API", version="1.0")

# Define the expected JSON structure
class TextInput(BaseModel):
    text: str
    max_length: int = 130
    min_length: int = 30

@app.get("/")
def root():
    return {"message": "Welcome to the AI Text Summarizer API. Please POST JSON data to summarize with fields: text, max_length, min_length."}

@app.post("/summarize")
def summarize_text(data: TextInput):
    if not data.text.strip():
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    try:
        summary = generate_summary(
            data.text,
            max_length=data.max_length,
            min_length=data.min_length
        )
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during summarization: {str(e)}")