# 🛡️ SENTINEL 2.0 (Formerly DrishTI) - Gemma Hackathon

SENTINEL is an agentic, on-device security orchestrator that bridges computer vision, audio processing, and Local LLM reasoning to detect, investigate, and escalate emergencies autonomously.

Built entirely around **Gemma**, it features a **Three-Tier Architecture**:
1. **Tier-1 Vision (Reflex)**: A low-latency YOLOv8 computer vision loop that detects candidate events (e.g., someone falling on camera). It also applies real-time HIPAA-compliant face pixelation and streams MJPEG video.
2. **Tier-2 Audio & STT (Sensory)**: Faster-Whisper + SpeechBrain SepFormer for isolating voices from noisy environments and transcribing audio automatically on anomaly detection.
3. **Tier-3 Orchestrator (Brain)**: The Gemma agent actively uses tools to decide if an alert is benign or critical, securing the decision in an immutable SQLite ledger. (Highly optimized: Gemma is restricted to exactly 1 tool call for blazing-fast hackathon response times, followed by immediate Twilio SMS dispatch).

---

## 🖥️ Mission Control Dashboard
Sentinel comes with a fully customized **Glassmorphism Mission Control Dashboard**.
- **Live Feeds**: View 3 zero-latency MJPEG feeds simultaneously (Raw, YOLO, Privacy pixelation mask).
- **Live Streaming Logs**: The dashboard uses a 500ms polling endpoint to stream Gemma's thoughts, tool calls, and decisions in exactly real-time without blocking on the database!
- **Auto-Extracted Audio**: When a fall is detected, ambient audio is extracted in milliseconds and injected into both the UI and the Twilio SMS.
- **Two-Way Audio Alerts**: The UI tracks when Gemma activates the physical speakers to talk to subjects.

---

## 🍏 Getting Started on MacOS (Apple Silicon / Intel)

This guide is meant for the team to get the repository running cleanly on Mac.

### 1. Prerequisites
You will need `ffmpeg` installed for the PyAV audio extraction and `ollama` for the LLM.
```bash
# Install FFmpeg (Required for audio extraction)
brew install ffmpeg

# Install Ollama (if you haven't already)
brew install --cask ollama
```

Make sure Ollama is running, then pull the model:
```bash
ollama run gemma4:e4b
```

### 2. Python Environment Setup
We recommend using a Python 3.10+ virtual environment.
```bash
python3 -m venv venv
source venv/bin/activate
```

Install the dependencies:
```bash
pip install -r requirements.txt
```
*(Note: If you run into issues with `torchcodec` on Mac, the code already gracefully falls back to `PyAV` for extracting audio from video files!)*

---

## 🚀 Running the End-to-End Story

To run the complete demo for the judges, you need to launch the architecture's three microservices in **three separate terminals**. 

Make sure your virtual environment is activated in all three!

### Terminal 1: The Host Server & Dashboard
This starts the backend API and serves the frontend dashboard.
```bash
python host_server.py
```
*(Runs on `http://127.0.0.1:8000`)*

### Terminal 2: The Gemma Agent & Audio Engine
This starts the heavy lifting STT/Audio separation engine and the main LLM entry point.
```bash
python main.py
```
*(Runs on `http://127.0.0.1:5000`)*

### Terminal 3: The Vision Trigger
This starts the video processing loop, MJPEG streams, and candidate event trigger.
Make sure you use the compiled video that has the audio stitched in!
```bash
python vision_trigger.py --source "Fall2_Cam4_with_audio.mp4"
```
*(Runs on `http://127.0.0.1:5001`)*

### 🎮 The Grand Finale
Once all three terminals are running, open your browser to:
**[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

You'll see the Mission Control UI. Turn your speakers up! When the patient falls, you'll hear Gemma use the Two-Way audio (`pyttsx3`) to ask if they need help, analyze their response using Whisper, and automatically escalate the situation. 

---

## 🛠️ Key Files
- `orchestrator.py`: The Gemma LLM agent tool-calling loop and 3-strike safety net.
- `vision_trigger.py`: The Tier-1 YOLO vision script with Face Blurring and MJPEG streams.
- `audio_engine.py`: Faster-Whisper and SepFormer initialization.
- `host_server.py`: FastAPI server bridging the UI, Database Ledger, and Orchestrator.
- `templates/index.html`: The Glassmorphism UI.
