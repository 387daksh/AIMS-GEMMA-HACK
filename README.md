# 🛡️ SENTINEL 2.0 (Formerly DrishTI) - Gemma Hackathon

SENTINEL is an agentic, on-device security orchestrator that bridges computer vision, audio processing, and Local LLM reasoning to detect, investigate, and escalate emergencies autonomously.

Built entirely around **Gemma**, it features a **Three-Tier Architecture**:
1. **Tier-1 Vision (Reflex)**: A low-latency YOLOv8 computer vision loop that detects candidate events (e.g., someone falling on camera). It also applies real-time HIPAA-compliant face pixelation and streams MJPEG video.
2. **Tier-2 Audio & STT (Sensory)**: Faster-Whisper + SpeechBrain SepFormer for isolating voices from noisy environments and transcribing audio automatically on anomaly detection.
3. **Tier-3 Orchestrator (Brain)**: The Gemma agent actively uses tools to decide if an alert is benign or critical, securing the decision in an immutable SQLite ledger. (Highly optimized: Gemma is restricted to exactly 1 tool call for blazing-fast hackathon response times, followed by immediate Twilio SMS dispatch).

---

## 🖥️ Mission Control Dashboard & Wow Factors
Sentinel comes with a fully customized **Glassmorphism Mission Control Dashboard**.
- **Live "Fake Dots" Skeletal Tracking**: The vision pipeline uses `yolov8n-pose.pt` to actively map the human body geometry during a fall. These skeletal maps are piped directly into the UI dashboard when Gemma uses the `zoom` or `recheck` tools!
- **Hacker Typewriter UI**: Watch Gemma's thoughts and tool-calls physically type out on the screen in real time as she actively investigates the scene.
- **Auto-Speak "Voice of God"**: The instant a fall is detected, the machine physically speaks out loud using `pyttsx3` ("Warning. Fall detected. Sentinel AI is analyzing the scene.") without blocking the LLM reasoning pipeline.
- **Live CCTV Scanline Feeds**: View 3 zero-latency MJPEG feeds simultaneously (Raw, YOLO Pose, Privacy pixelation mask) overlaid with an authentic CRT scanline effect.
- **Manual Override Button**: A dedicated UI button to instantly bypass the AI and trigger a Twilio SMS escalation.
- **Hard Demo Lockout**: Prevents the UI from clearing during a presentation once a Critical Alert is reached.

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

To run the complete demo for the judges, you no longer need to open three terminals! We built a master launch script.

Make sure your virtual environment is activated, then simply run:

```bash
python run_sentinel.py
```

This single command will spin up all three microservices simultaneously:
1. **The Host Server** (`http://127.0.0.1:8000`)
2. **The Gemma Agent** (`http://127.0.0.1:5000`)
3. **The Vision Trigger** (`http://127.0.0.1:5001`)

### 🎮 The Grand Finale
Once the script prints `✅ ALL SYSTEMS ONLINE!`, open your browser to:
**[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

You'll see the Mission Control UI. Turn your speakers up! When the patient falls, the laptop will announce the detection out loud, the CCTV feeds will show the skeletal tracking, and you'll watch Gemma actively type out her investigation in the terminal before dispatching the Twilio SMS alert. 

---

## 🛠️ Key Files
- `orchestrator.py`: The Gemma LLM agent tool-calling loop and 3-strike safety net.
- `vision_trigger.py`: The Tier-1 YOLO vision script with Face Blurring and MJPEG streams.
- `audio_engine.py`: Faster-Whisper and SepFormer initialization.
- `host_server.py`: FastAPI server bridging the UI, Database Ledger, and Orchestrator.
- `templates/index.html`: The Glassmorphism UI.
