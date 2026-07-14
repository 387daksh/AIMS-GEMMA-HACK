import os
import shutil
from fastapi import FastAPI, File, UploadFile
import uvicorn

# Import the team's newly pulled audio engine!
import audio_engine

app = FastAPI(title="SENTINEL 2.0 Host Server")

# A simple state tracker to mock a multi-turn investigation by Gemma 4
# Once you have Gemma 4 running locally, you replace this function with a real LLM call.
call_count = 0

@app.post("/analyze")
def analyze(payload: dict):
    global call_count
    call_count += 1
    
    # Turn 1: Gemma asks to clean the noisy audio
    if call_count == 1:
        return {
            "tool_call": "process_audio",
            "tool_args": {"file_path": "dummy_audio.wav"}
        }
    
    # Turn 2: Gemma asks for fresh frames after a short delay
    if call_count == 2:
        return {
            "tool_call": "recheck",
            "tool_args": {"after_seconds": 2}
        }
    
    # Turn 3: Gemma makes a final decision and raises an alert
    call_count = 0  # reset for next run
    return {
        "action": "raise_alert",
        "args": {
            "severity": "CRITICAL",
            "justification": "Person fell to the ground. Cleaned audio reveals a distress scream. No recovery in follow-up frames.",
            "evidence_ids": ["audio_transcript_1", "recheck_frames_4"]
        }
    }

@app.post("/process-audio")
async def process_audio(audio_file: UploadFile = File(...)):
    """
    Receives noisy audio from Daksh's Orchestrator, saves it, and runs
    the team's SpeechBrain (SepFormer) audio_engine to clean it.
    """
    input_path = f"temp_in_{audio_file.filename}"
    output_path = f"temp_out_cleaned_{audio_file.filename}"
    
    # 1. Save the incoming payload to disk
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(audio_file.file, buffer)
        
    print(f"🔊 Processing audio with SepFormer: {input_path}")
    
    # 2. Run the team's actual SpeechBrain SepFormer separation!
    cleaned_file_path = audio_engine.clean_audio(input_path, output_path)
    print(f"✅ Audio cleaned and saved to {cleaned_file_path}")
    
    # 3. In a 100% complete pipeline, this cleaned wav would go to Whisper or Gemma-Audio.
    # We simulate the text output for the demo:
    simulated_transcript = "[DISTRESS DETECTED] Help me!"
    
    # Clean up the noisy temp file
    if os.path.exists(input_path):
        os.remove(input_path)
    
    return {
        "cleaned_text": simulated_transcript, 
        "cleaned_file_path": cleaned_file_path
    }

@app.post("/recheck")
def recheck(payload: dict):
    # Mocking fresh frame check
    return {"result": "Subject remains motionless on the ground."}

@app.post("/zoom")
def zoom(payload: dict):
    return {"result": "Zoom clear."}

@app.post("/get-history")
def get_history(payload: dict):
    return {"result": "No prior incidents in the last 60 minutes."}

if __name__ == "__main__":
    print("🚀 Starting Syna's Host Server on port 5000...")
    uvicorn.run(app, host="127.0.0.1", port=5000)
