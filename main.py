import requests
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
# 1. Import Syna's newly refactored AudioEngine class
from audio_engine import AudioEngine

# Initialize the SpeechBrain engine on startup so it doesn't reload on every request
engine = AudioEngine()

app = FastAPI()
OLLAMA_URL = "http://localhost:11434/api/generate"

class PromptPayload(BaseModel):
    prompt: str

@app.post("/analyze")
async def analyze_text(payload: PromptPayload):
    # Forwards Daksh's prompt directly to your background Ollama engine
    response = requests.post(OLLAMA_URL, json={
        "model": "gemma4",
        "prompt": payload.prompt,
        "stream": False
    })
    return response.json()

@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    # Saves incoming audio from Daksh's loop
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    # 2. Map directly to Syna's new class method
    clean_track_path = engine.isolate_audio(temp_path)
    
    return {"status": "success", "cleaned_file": clean_track_path}