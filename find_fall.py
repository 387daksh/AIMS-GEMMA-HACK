import cv2

def find_fall_time(video_path):
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    from ultralytics import YOLO
    model = YOLO("yolov8n.pt")
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        results = model(frame, verbose=False)[0]
        
        for box in results.boxes:
            class_id = int(box.cls[0])
            if class_id == 0:  # person
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                w = x2 - x1
                h = y2 - y1
                if h > 0 and (w / h) > 1.2:
                    time_sec = frame_count / fps
                    print(f"[*] Fall detected at {time_sec:.2f} seconds (Frame {frame_count})")
                    return time_sec
        frame_count += 1
    
    print("[-] No fall detected.")
    return None

if __name__ == "__main__":
    find_fall_time(r"E:\gemma hack\Fall2_Cam4.mp4")
