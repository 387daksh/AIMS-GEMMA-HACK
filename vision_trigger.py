import argparse
import base64
import os
import threading
import time
from collections import deque
from typing import Optional

import cv2
import numpy as np
import requests
from ultralytics import YOLO
from fastapi import FastAPI, HTTPException
from scipy.io import wavfile
import uvicorn

# --- CONFIG (all localhost, all overridable via env for the demo) ---
ORCHESTRATOR_URL = os.getenv("SENTINEL_ORCHESTRATOR_URL", "http://127.0.0.1:8000/candidate-event")
TOOL_SERVER_PORT = int(os.getenv("SENTINEL_VISION_TOOL_PORT", "5001"))
CAMERA_ID = os.getenv("SENTINEL_CAMERA_ID", "Cam-MacBook-Air")

COOLDOWN = float(os.getenv("SENTINEL_EVENT_COOLDOWN", "8"))  # min seconds between candidate-events per camera
MOTION_THRESHOLD = float(os.getenv("SENTINEL_MOTION_THRESHOLD", "0.18"))  # below this, treat as trivial jitter
FRAME_BUFFER_MAXLEN = 200  # ~7-13s of raw full-res frames depending on fps; keeps memory bounded

# Load an ultra-lightweight object detector (downloads automatically on first run)
model = YOLO("yolov8n.pt")

# --- SHARED STATE (written by the capture loop, read by the tool server thread) ---
buffer_lock = threading.Lock()
frame_buffer = deque(maxlen=FRAME_BUFFER_MAXLEN)  # entries: (ts, raw_bgr_frame, boxes, confidences, motion_score)
trace_buffer = deque(maxlen=1000)  # lightweight historical trace: (ts, boxes, motion_score)
audio_state = {"samples": None, "sample_rate": None, "session_start": None}


def parse_args():
    parser = argparse.ArgumentParser(description="SENTINEL Tier-1 vision trigger")
    parser.add_argument(
        "--source",
        default=os.getenv("SENTINEL_VIDEO_SOURCE", "0"),
        help="Webcam index (e.g. 0) or path to a staged video file for testing",
    )
    parser.add_argument("--headless", action="store_true", help="Skip cv2.imshow (useful when driving a staged file unattended)")
    return parser.parse_args()


def resolve_source(source: str):
    """Webcam index strings ('0', '1', ...) become ints; anything else is a file path."""
    if source.isdigit():
        return int(source), False
    return source, True


def load_audio_track(path: str):
    """
    Staged test clips carry their own audio track — extract it once at
    startup into an in-memory buffer that /audio can slice from. Live
    webcam capture has no microphone pipeline wired up, so audio_state
    stays empty in that mode and get_audio degrades gracefully.
    """
    try:
        import torchaudio
        waveform, sr = torchaudio.load(path)
        mono = waveform.mean(dim=0).numpy().astype(np.float32)
        audio_state["samples"] = mono
        audio_state["sample_rate"] = sr
        print(f"[+] Loaded audio track from {path}: {mono.shape[0] / sr:.1f}s @ {sr}Hz")
    except Exception as e:
        print(f"[-] No usable audio track in '{path}' ({e}). get_audio will report unavailable.")


# ---------------------------------------------------------------------------
# Tool server: lets the orchestrator genuinely re-observe this camera after
# the fact (recheck), crop the RAW full-res frame (zoom), and pull a slice
# of the rolling audio buffer (get_audio). Runs in a background thread
# alongside the capture loop so this stays a single process.
# ---------------------------------------------------------------------------
tool_app = FastAPI(title="SENTINEL Vision Tool Server")


def _encode_jpg(frame) -> str:
    ok, buf = cv2.imencode(".jpg", frame)
    if not ok:
        raise HTTPException(500, "Failed to encode frame as JPEG")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def _nearest_frame(frame_ts: Optional[float]):
    with buffer_lock:
        if not frame_buffer:
            return None
        if frame_ts is None:
            return frame_buffer[-1]
        return min(frame_buffer, key=lambda item: abs(item[0] - frame_ts))


@tool_app.get("/health")
def health():
    with buffer_lock:
        buffered = len(frame_buffer)
    return {"status": "ok", "camera_id": CAMERA_ID, "buffered_frames": buffered}


@tool_app.get("/recheck")
def recheck(after_seconds: float = 2.0):
    """Waits, then returns whatever the capture loop has actually seen since — a real re-observation, not a replay."""
    after_seconds = max(0.0, min(after_seconds, 15.0))
    time.sleep(after_seconds)
    with buffer_lock:
        if not frame_buffer:
            raise HTTPException(503, "No frames buffered yet")
        ts, frame, boxes, confidences, motion_score = frame_buffer[-1]
    return {
        "camera_id": CAMERA_ID,
        "frame_ts": ts,
        "waited_seconds": after_seconds,
        "person_count": len(boxes),
        "boxes": boxes,
        "confidences": confidences,
        "motion_score": motion_score,
        "frame_jpg_base64": _encode_jpg(frame),
    }


@tool_app.get("/trace")
def get_trace(seconds_back: float = 30.0):
    now = time.time()
    cutoff = now - seconds_back
    trace_data = []
    with buffer_lock:
        last_ts = 0
        for ts, boxes, motion in trace_buffer:
            if ts >= cutoff and (ts - last_ts >= 0.5):
                trace_data.append({
                    "ts": round(ts, 2),
                    "boxes": boxes,
                    "motion_score": motion
                })
                last_ts = ts
    return {"status": "ok", "camera_id": CAMERA_ID, "trace": trace_data}


@tool_app.get("/zoom")
def zoom(x1: float, y1: float, x2: float, y2: float, frame_ts: Optional[float] = None):
    """Crops from the RAW full-resolution buffered frame, not the YOLO inference copy."""
    item = _nearest_frame(frame_ts)
    if item is None:
        raise HTTPException(503, "No frames buffered yet")
    ts, frame, *_ = item
    h, w = frame.shape[:2]
    ix1, iy1 = max(0, int(x1)), max(0, int(y1))
    ix2, iy2 = min(w, int(x2)), min(h, int(y2))
    if ix2 <= ix1 or iy2 <= iy1:
        raise HTTPException(400, "Invalid crop box")
    crop = frame[iy1:iy2, ix1:ix2]
    return {
        "camera_id": CAMERA_ID,
        "frame_ts": ts,
        "box": [ix1, iy1, ix2, iy2],
        "crop_jpg_base64": _encode_jpg(crop),
    }


@tool_app.get("/audio")
def get_audio_clip(t_start: float, t_end: float):
    """
    Slices the rolling audio buffer by wall-clock window and returns acoustic
    features (peak/RMS) plus the raw clip path. No transcript is produced —
    there is no speech-to-text model in this stack, only SepFormer source
    separation (see /isolate). Gemma reasons over the acoustic features and
    decides whether to call isolate_audio.
    """
    samples = audio_state["samples"]
    sr = audio_state["sample_rate"]
    session_start = audio_state["session_start"]
    if samples is None or session_start is None:
        raise HTTPException(503, "No audio buffer available (live webcam mode has no microphone capture wired up)")

    rel_start = max(0.0, t_start - session_start)
    rel_end = max(rel_start, t_end - session_start)
    i_start = int(rel_start * sr)
    i_end = min(len(samples), int(rel_end * sr))
    if i_start >= i_end:
        raise HTTPException(400, "Requested audio window is empty or out of range")

    clip = samples[i_start:i_end]
    peak = float(np.max(np.abs(clip))) if clip.size else 0.0
    rms = float(np.sqrt(np.mean(clip ** 2))) if clip.size else 0.0

    os.makedirs("audio_clips", exist_ok=True)
    clip_path = os.path.join("audio_clips", f"clip_{int(t_start * 1000)}_{int(t_end * 1000)}.wav")
    wavfile.write(clip_path, sr, np.clip(clip * 32767, -32768, 32767).astype(np.int16))

    return {
        "camera_id": CAMERA_ID,
        "clip_path": os.path.abspath(clip_path),
        "sample_rate": sr,
        "duration_seconds": round(len(clip) / sr, 3),
        "peak_amplitude": round(peak, 4),
        "rms_amplitude": round(rms, 4),
    }


def run_tool_server():
    config = uvicorn.Config(tool_app, host="127.0.0.1", port=TOOL_SERVER_PORT, log_level="warning")
    uvicorn.Server(config).run()


# ---------------------------------------------------------------------------
# Tier-1 capture loop
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    source, is_file = resolve_source(args.source)

    threading.Thread(target=run_tool_server, daemon=True).start()
    print(f"[*] Vision tool server listening on http://127.0.0.1:{TOOL_SERVER_PORT} (recheck/zoom/audio)")

    session_start = time.time()
    audio_state["session_start"] = session_start
    if is_file:
        load_audio_track(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video source: {source}")

    # For a staged file, pace reads to roughly real-time so recheck's wall-clock
    # wait actually corresponds to new footage, same as it would from a live camera.
    source_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_interval = (1.0 / source_fps) / 4.0 if is_file and source_fps > 0 else 0.0

    last_sent_by_camera = {}
    prev_gray_frame = None
    loop_count = 0
    frame_counter = 0
    
    # Store the last YOLO output to reuse during skipped frames
    last_persons = []
    last_boxes = []
    last_annotated = None

    print("SENTINEL Tier-1 Reflex Active. Monitoring feed... (Press 'q' in the video window to quit)")

    while cap.isOpened():
        frame_counter += 1
        loop_start = time.time()
        ret, frame = cap.read()
        if not ret:
            if is_file:
                # Keep the buffer/tool server (and any in-flight investigation reading
                # from it) alive indefinitely for file-source testing, instead of
                # exiting at EOF — a slow episode should never outlive its own camera.
                loop_count += 1
                print(f"[*] Reached end of staged file, looping back to start (loop #{loop_count})...")
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                prev_gray_frame = None  # avoid a spurious motion spike across the last-frame -> first-frame jump
                audio_state["session_start"] = time.time()  # reset audio sync
                continue
            break

        now = time.time()

        # --- REAL MOTION SCORE (frame differencing) ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        if prev_gray_frame is None:
            motion_score = 0.0
        else:
            frame_delta = cv2.absdiff(prev_gray_frame, gray)
            motion_score = min(frame_delta.mean() / 25.0, 1.0)

        prev_gray_frame = gray

        # --- YOLO DETECTION (throttled to every 3rd frame to save CPU) ---
        if frame_counter % 3 == 0 or not last_boxes:
            results = model(frame, verbose=False)[0]
            persons = []
            boxes = []
            for box in results.boxes:
                class_id = int(box.cls[0])
                if class_id == 0:  # COCO index 0 is 'person'
                    confidence = float(box.conf[0])
                    xyxy = box.xyxy[0].tolist()
                    persons.append(confidence)
                    boxes.append(xyxy)
            last_persons = persons
            last_boxes = boxes
            if not args.headless:
                last_annotated = results.plot()
        else:
            persons = last_persons
            boxes = last_boxes

        with buffer_lock:
            frame_buffer.append((now, frame.copy(), boxes, persons, round(motion_score, 3)))
            trace_buffer.append((now, boxes, round(motion_score, 3)))

        person_detected = len(persons) > 0
        top_confidence = max(persons) if persons else 0.0
        last_sent = last_sent_by_camera.get(CAMERA_ID, 0)
        
        # Check aspect ratio for falls (width > height * 1.2)
        is_fallen = False
        for box in boxes:
            w = box[2] - box[0]
            h = box[3] - box[1]
            if h > 0 and (w / h) > 1.2:
                is_fallen = True
                break

        # Gate on BOTH a confident person detection AND real motion, not just presence —
        # a person sitting still in frame was re-firing every COOLDOWN seconds forever
        # since motion_score was computed but never actually checked before.
        if (person_detected and top_confidence > 0.60
                and (motion_score >= MOTION_THRESHOLD or is_fallen)
                and (now - last_sent) > COOLDOWN):
            print(f"Anomaly spotted! {len(persons)} person(s) detected (top confidence: {top_confidence:.2f}, motion: {motion_score:.2f})")
            payload = {
                "timestamp": now,
                "camera_id": CAMERA_ID,
                "motion_score": round(motion_score, 3),
                "vision_confidence": top_confidence,
                "persons": persons,
                "boxes": boxes,
                "fallen": is_fallen,
                "event_type": "fall_detected" if is_fallen else "person_spot",
                "tool_server_url": f"http://127.0.0.1:{TOOL_SERVER_PORT}",
            }
            try:
                response = requests.post(ORCHESTRATOR_URL, json=payload, timeout=3)
                print(f"Handshake sent to orchestrator. Server response status: {response.status_code}")
            except Exception as e:
                print(f"Network transmission delayed (Server offline or URL incorrect): {e}")

            last_sent_by_camera[CAMERA_ID] = now

        if not args.headless:
            if last_annotated is not None:
                cv2.imshow("SENTINEL - Eyes", last_annotated)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        if frame_interval:
            elapsed = time.time() - loop_start
            remaining = frame_interval - elapsed
            if remaining > 0:
                time.sleep(remaining)

    cap.release()
    cv2.destroyAllWindows()
    print("SENTINEL Eyes shut down.")


if __name__ == "__main__":
    main()
