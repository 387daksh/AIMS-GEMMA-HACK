import requests
import time

# Create a tiny 1x1 black pixel in base64 to simulate an image payload
dummy_image = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="

payload = {
    "timestamp": time.time(),
    "camera_id": "Cam-Mock-01",
    "motion_score": 0.92,
    "vision_confidence": 0.88,
    "persons": [0.88],
    "boxes": [[100, 150, 400, 500]],
    "event_type": "person_spot",
    "image_base64": dummy_image,
    "local_audio_buffer": "sample_noise.wav"
}

print("Firing mock camera event...")
try:
    resp = requests.post("http://127.0.0.1:8000/candidate_event", json=payload, timeout=5)
    print("Orchestrator responded:", resp.json())
except Exception as e:
    print("Error hitting orchestrator:", e)
