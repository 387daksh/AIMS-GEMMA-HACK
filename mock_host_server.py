from fastapi import FastAPI, File, UploadFile
import uvicorn

app = FastAPI(title="Mock SENTINEL Host Server")

# A simple state tracker to mock a multi-turn investigation by Gemma 4
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
            "justification": "Person fell to the ground. Cleaned audio reveals a scream. No recovery in follow-up frames.",
            "evidence_ids": ["audio_transcript_1", "recheck_frames_4"]
        }
    }

@app.post("/process-audio")
async def process_audio(audio_file: UploadFile = File(...)):
    # Mocking SepFormer audio cleanup
    return {"cleaned_text": "[DISTRESS DETECTED] Help me!"}

@app.post("/recheck")
def recheck(payload: dict):
    # Mocking fresh frame check
    return {"result": "Subject remains motionless on the ground."}

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=5000)
