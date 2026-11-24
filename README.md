# SENTINEL (formerly DrishTI)

**🥉 3rd place, Google Gemma Hackathon (July 2026)**

An on-device AI safety investigator. A camera spots a possible emergency (for example, someone falling), and a local Gemma agent investigates the scene, decides whether it is real, and escalates by SMS. Video and reasoning stay on the machine; only the alert leaves it.

`Gemma 4 (Ollama)` `YOLOv8-pose` `Faster-Whisper` `SpeechBrain SepFormer` `FastAPI` `SQLite` `Twilio`

## How it works

```
Camera ──> Tier 1: Vision (reflex)      YOLOv8-pose flags a candidate event, pixelates faces
               │
               ▼
Mic    ──> Tier 2: Audio (senses)       SepFormer isolates voices, Faster-Whisper transcribes
               │
               ▼
           Tier 3: Gemma agent (brain)  calls tools (zoom, recheck) to investigate,
                                        classifies benign vs critical
               │
               ▼
           Hash-chained SQLite ledger + Twilio SMS escalation
```

- **Vision:** low-latency YOLOv8 loop with pose skeletons and real-time face pixelation for privacy; three live MJPEG feeds (raw, pose, privacy mask).
- **Audio:** voice isolation in noisy rooms, transcription triggered only on anomalies.
- **Agent:** Gemma decides with tool calls, capped for fast response; every decision is written to a hash-chained, signed SQLite ledger.
- **Dashboard:** live feeds, the agent's reasoning streamed as it investigates, spoken alert on detection, and a manual override that escalates instantly.

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
