import cv2
import time
from ultralytics import YOLO

# Load an ultra-lightweight object detector
model = YOLO("yolov8n.pt")

# Hardcoded to your test video file
VIDEO_SOURCE = "E:\\gemma hack\\Fall2_Cam4.mp4"

# Connect to the chosen video source
cap = cv2.VideoCapture(VIDEO_SOURCE)

# Keep track of the previous frame so we can measure real motion
prev_gray_frame = None

print("🎥 Testing YOLO on video file... (Press 'q' in the video window to quit)")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("🎬 End of video reached.")
        break

    # --- REAL MOTION SCORE (frame differencing) ---
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_gray_frame is None:
        motion_score = 0.0
    else:
        frame_delta = cv2.absdiff(prev_gray_frame, gray)
        # Normalize to a 0-1 range
        motion_score = min(frame_delta.mean() / 25.0, 1.0)

    prev_gray_frame = gray

    # --- YOLO DETECTION ---
    results = model(frame, verbose=False)[0]
    persons = []
    
    for box in results.boxes:
        class_id = int(box.cls[0])
        if class_id == 0:  # COCO index 0 is 'person'
            confidence = float(box.conf[0])
            persons.append(confidence)

    person_detected = len(persons) > 0
    top_confidence = max(persons) if persons else 0.0

    # Draw detection boxes on the frame so you can SEE what YOLO sees
    annotated_frame = results.plot()

    # Show the live window
    cv2.imshow("SENTINEL - Video Test", annotated_frame)

    # Quit cleanly by pressing 'q' while the video window is focused
    # Note: waitKey(30) slows down playback to ~30fps so it doesn't fly by instantly!
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Test shut down.")
