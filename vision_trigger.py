import cv2
import requests
import time
from ultralytics import YOLO

# Load an ultra-lightweight object detector (downloads automatically on first run)
model = YOLO("yolov8n.pt")

# Target Ayush's network server link (Swap this URL once Ayush gives you his ngrok link!)
AYUSH_SERVER_URL = "https://CHANGE_THIS_TO_YOUR_TEAM_NGROK_URL.ngrok-free.app/candidate_event"

# Connect to your MacBook Air's built-in webcam
cap = cv2.VideoCapture(0)

# Non-blocking cooldown tracker (replaces time.sleep)
last_sent = 0
COOLDOWN = 5  # seconds between event sends, so you don't spam the network

print("🛡️ SENTINEL Tier-1 Reflex Active. Monitoring camera feed... (Press 'q' in the video window to quit)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Sub-second inference tracking on the frame
    results = model(frame, verbose=False)[0]
    person_detected = False
    confidence = 0.0

    # Parse detected boxes to find humans
    for box in results.boxes:
        class_id = int(box.cls[0])
        if class_id == 0:  # COCO index 0 is explicitly 'person'
            person_detected = True
            confidence = float(box.conf[0])
            break

    # If a person is spotted with clear confidence, and cooldown has passed, send event
    if person_detected and confidence > 0.60 and (time.time() - last_sent) > COOLDOWN:
        print(f"⚠️ Anomaly spotted! Human detected (Confidence: {confidence:.2f})")
        payload = {
            "timestamp": time.time(),
            "camera_id": "Cam-MacBook-Air",
            "motion_score": 0.85,
            "vision_confidence": confidence,
            "event_type": "person_spot"
        }
        try:
            response = requests.post(AYUSH_SERVER_URL, json=payload, timeout=3)
            print(f"📡 Handshake sent to controller loop. Server response status: {response.status_code}")
        except Exception as e:
            print(f"❌ Network transmission delayed (Server offline or URL incorrect): {e}")

        last_sent = time.time()  # reset cooldown timer, loop keeps running

    # Draw detection boxes on the frame so you can SEE what YOLO sees
    annotated_frame = results.plot()

    # Show the live window
    cv2.imshow("SENTINEL - Eyes", annotated_frame)

    # Quit cleanly by pressing 'q' while the video window is focused
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 SENTINEL Eyes shut down.")