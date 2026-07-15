# SENTINEL 2.0: Agentic AI Security System

SENTINEL is a multi-agent AI security pipeline powered by Gemma. It utilizes a vision trigger (YOLOv8) to detect anomalies (like falls) and orchestrates a local LLM to reason about the event, call tools (Zoom, Recheck, Audio Isolation + Transcription), and securely log terminal decisions to a cryptographic SQLite ledger.

## Requirements for Mac
- **Python 3.10+**
- **Ollama** (Native Mac App): [Download Ollama](https://ollama.com/)
- **Git**

## Quick Start (Mac)

### 1. Set Up Ollama (The AI Brain)
Download and install the native Ollama app for Mac. Once installed, pull the Gemma model:
```bash
ollama run gemma:2b
```
*(Leave this running in the background, or ensure the Ollama menu bar icon is active).*

### 2. Set Up the Python Environment
Open your terminal and clone the repository. Then set up a virtual environment and install the required dependencies:
```bash
python3 -m venv env
source env/bin/activate

# Install the core ML and Web libraries
pip install fastapi uvicorn requests opencv-python ultralytics speechbrain faster-whisper
```

### 3. Run the Microservices
SENTINEL is composed of 3 distinct services. Open **3 separate terminal tabs**, ensure your virtual environment is activated in all of them (`source env/bin/activate`), and run them in this order:

**Terminal 1: Audio Service (Speech-to-Text & Noise Isolation)**
```bash
python main.py
```

**Terminal 2: Agent Orchestrator (The "Brain")**
```bash
python host_server.py
```

**Terminal 3: Vision Trigger (The "Eyes")**
```bash
# Run the camera trigger on your video file or webcam
python vision_trigger.py --source "Fall2_Cam4.mp4"
```

### 4. Observe the Multi-Agent Loop
When the Vision Trigger spots an anomaly (like a person falling), it will trigger the Orchestrator. Switch to **Terminal 2** to watch Gemma think in real-time as it:
1. Rechecks the camera feed.
2. Isolates and transcribes the audio using `faster-whisper`.
3. Makes a final decision (`raise_alert` or `log_benign`).
4. Cryptographically seals the record into `ledger.db`.

You can view the raw logs in `anchor.log`.

## Troubleshooting
- **Port Conflicts**: Ensure ports `8000` and `5000` are free before starting.
- **Missing Audio Track**: If you see `TorchCodec is required`, it just means the video file has no audio track. The pipeline will still function purely on vision.
