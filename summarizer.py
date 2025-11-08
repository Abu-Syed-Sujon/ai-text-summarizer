from transformers import pipeline
import os
from dotenv import load_dotenv

load_dotenv()

# Set default model, can also set via .env file
MODEL_NAME = os.getenv("MODEL_NAME", "sshleifer/distilbart-cnn-12-6")

# Initialize summarization pipeline
summarizer = pipeline("summarization", model=MODEL_NAME)

def generate_summary(text: str, max_length: int = 130, min_length: int = 30) -> str:
    # Check for empty input
    if not text.strip():
        return "Error: Input text is empty."

    # Limit text length to prevent crashes
    if len(text) > 5000:
        text = text[:5000]

    try:
        summary = summarizer(
            text,
            max_length=max_length,
            min_length=min_length,
            do_sample=False
        )
        return summary[0]["summary_text"]
    except Exception as e:
        return f"Error during summarization: {str(e)}"
