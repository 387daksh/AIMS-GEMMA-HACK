import requests
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
# 1. Import Syna's standalone function directly
from audio_engine import clean_audio 

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
    
    # 2. Map directly to Syna's function signature (input_path, output_path)
    clean_track_path = f"processed_{file.filename}"
    clean_audio(temp_path, clean_track_path)
    
    return {"status": "success", "cleaned_file": clean_track_path}