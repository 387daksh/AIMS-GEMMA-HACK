import cv2
import requests
import time
from ultralytics import YOLO

# Load an ultra-lightweight object detector (downloads automatically on first run)
model = YOLO("yolov8n.pt")

# Points to Daksh's controller loop now (he'll send you his ngrok link once his
# server is up — paste it here).
DAKSH_SERVER_URL = "http://127.0.0.1:8000/candidate_event"

# Connect to your webcam
cap = cv2.VideoCapture(0)

# Non-blocking cooldown tracker (replaces time.sleep)
last_sent = 0
COOLDOWN = 5  # seconds between event sends, so you don't spam the network

# Keep track of the previous frame so we can measure real motion
prev_gray_frame = None

print("🛡️ SENTINEL Tier-1 Reflex Active. Monitoring camera feed... (Press 'q' in the video window to quit)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- REAL MOTION SCORE (frame differencing) ---
    # Convert to grayscale + blur a little, then compare against the last frame.
    # This turns "did anything move" into an actual number instead of a hardcoded 0.85.
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_gray_frame is None:
        motion_score = 0.0
    else:
        frame_delta = cv2.absdiff(prev_gray_frame, gray)
        # Normalize to a 0-1 range so it's comparable across cameras/lighting
        motion_score = min(frame_delta.mean() / 25.0, 1.0)

    prev_gray_frame = gray

    # --- YOLO DETECTION (now collects EVERY person, not just the first) ---
    results = model(frame, verbose=False)[0]
    persons = []
    boxes = []

    for box in results.boxes:
        class_id = int(box.cls[0])
        if class_id == 0:  # COCO index 0 is 'person'
            confidence = float(box.conf[0])
            xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
            persons.append(confidence)
            boxes.append(xyxy)

    person_detected = len(persons) > 0
    top_confidence = max(persons) if persons else 0.0

    # If a person is spotted with clear confidence, and cooldown has passed, send event
    if person_detected and top_confidence > 0.60 and (time.time() - last_sent) > COOLDOWN:
        print(f"⚠️ Anomaly spotted! {len(persons)} person(s) detected (top confidence: {top_confidence:.2f}, motion: {motion_score:.2f})")
        payload = {
            "timestamp": time.time(),
            "camera_id": "Cam-MacBook-Air",
            "motion_score": round(motion_score, 3),
            "vision_confidence": top_confidence,
            "persons": persons,   # list of confidences, one per detected person
            "boxes": boxes,       # list of [x1, y1, x2, y2] boxes, one per person
            "event_type": "person_spot"
        }
        try:
            response = requests.post(DAKSH_SERVER_URL, json=payload, timeout=3)
            print(f"📡 Handshake sent to Daksh's controller loop. Server response status: {response.status_code}")
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
