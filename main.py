import requests
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from audio_engine import AudioEngine

app = FastAPI()
audio_engine = AudioEngine()
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
    # Saves incoming audio from Daksh's loop and cleans it
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())
    
    clean_track_path = audio_engine.isolate_audio(temp_path)
    return {"status": "success", "cleaned_file": clean_track_path}