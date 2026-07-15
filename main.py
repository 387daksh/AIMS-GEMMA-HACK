import os
from fastapi import FastAPI, UploadFile, File
import uvicorn
from audio_engine import AudioEngine

# Initialize the SpeechBrain engine once on startup so it doesn't reload per-request
engine = AudioEngine()

app = FastAPI(title="SENTINEL 2.0 Audio Service")


@app.post("/process-audio")
async def process_audio(file: UploadFile = File(...)):
    """
    Receives a raw/noisy audio clip, runs it through SepFormer, and returns
    the local file path of the isolated track.
    """
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    clean_track_path = engine.isolate_audio(temp_path)

    if os.path.exists(temp_path):
        os.remove(temp_path)

    return {"status": "success", "cleaned_file": clean_track_path}


if __name__ == "__main__":
    print("Starting SENTINEL Audio Service on port 5000...")
    uvicorn.run(app, host="127.0.0.1", port=5000)
