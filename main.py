import os
import uuid
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
    # Sanitize incoming file to prevent path traversal / disk leaks
    ext = os.path.splitext(file.filename)[1]
    safe_filename = f"{uuid.uuid4().hex}{ext}"
    temp_path = f"temp_{safe_filename}"
    
    with open(temp_path, "wb") as buffer:
        buffer.write(await file.read())

    import time
    start_time = time.time()
    
    # Isolate audio and dynamically pick the best channel
    res = engine.isolate_audio(temp_path)
    
    elapsed = time.time() - start_time
    print(f"[!] Audio isolation and STT pipeline completed in {elapsed:.2f} seconds.")
    
    transcript = ""
    if isinstance(res, tuple):
        clean_track_path, transcript = res
    else:
        clean_track_path = res

    # Clean up the raw uploaded temp file
    if os.path.exists(temp_path):
        os.remove(temp_path)
        
    # Clean up the isolated output file immediately to prevent unbounded disk growth
    # We only need to return the path string and the transcript to the client.
    if clean_track_path and not clean_track_path.startswith("Error") and os.path.exists(clean_track_path):
        os.remove(clean_track_path)

    return {"status": "success", "cleaned_file": clean_track_path, "transcript": transcript}


if __name__ == "__main__":
    print("Starting SENTINEL Audio Service on port 5000...")
    uvicorn.run(app, host="127.0.0.1", port=5000)
