import cv2
import time
import threading
import pyttsx3
import numpy as np
from ultralytics import YOLO
import winsound  

def play_beep():
    """Plays a short beep before voice alert (non-blocking)."""
    def _beep():
        frequency = 1000  
        duration = 200   
        winsound.Beep(frequency, duration)
    threading.Thread(target=_beep, daemon=True).start()

def speak(text):
    engine = pyttsx3.init()
    engine.setProperty("rate", 165)
    engine.setProperty("volume", 1.0)
    engine.say(text)
    engine.runAndWait()

def alert_user(text):
    """Plays beep + speaks message in background (non-blocking)."""
    def run():
        play_beep()
        speak(text)
    threading.Thread(target=run, daemon=True).start()

def get_position(box, width):
    """Return Left / Center / Right based on bounding box center."""
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2

    if cx < width / 3:
        return "left"
    elif cx > (2 * width / 3):
        return "right"
    else:
        return "center"

def get_movement_suggestion(position):
    """Suggest movement based on obstacle position."""
    if position == "left":
        return "move right"
    elif position == "right":
        return "move left"
    else:
        return "move back"

def get_alert_interval(box, frame_height):
    """Dynamic interval based on object size (closeness)."""
    x1, y1, x2, y2 = box
    height = y2 - y1
    rel_size = height / frame_height

    if rel_size > 0.5:  # Very close
        return 1.0
    elif rel_size > 0.3:  # Medium
        return 2.0
    else:  # Far
        return 3.0


def main():
    model = YOLO("yolov8n.pt")

   
    url = "http://10.114.137.85:8080/video"  
    cap = cv2.VideoCapture(url)

    if not cap.isOpened():
        print("❌ Could not open video stream. Check IP Webcam connection.")
        return

    last_alert_time = 0
    dynamic_interval = 2.5  # Default
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        
        frame = cv2.resize(frame, (320,240))
        H, W = frame.shape[:2]
        frame_count += 1

       
        if frame_count % 2 != 0:
            cv2.imshow("Obstacle Detector", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue

        results = model.predict(frame, conf=0.5, verbose=False)

        detections = []
        if results:
            r = results[0]
            for box in r.boxes:
                cls_id = int(box.cls)
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[cls_id]

                position = get_position((x1, y1, x2, y2), W)
                move = get_movement_suggestion(position)

                detections.append({
                    "label": label,
                    "xyxy": (x1, y1, x2, y2),
                    "position": position,
                    "suggestion": move
                })

        
        now = time.time()
        if detections:
          
            best = max(detections, key=lambda d: (d["xyxy"][3] - d["xyxy"][1]))
            dynamic_interval = get_alert_interval(best["xyxy"], H)

            if (now - last_alert_time) > dynamic_interval:
                phrase = f"{best['label']} ahead on {best['position']}, {best['suggestion']}."
                alert_user(phrase)
                last_alert_time = now

        # Draw detections
        for d in detections:
            x1, y1, x2, y2 = d["xyxy"]
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame,
                        f"{d['label']} ({d['position']})",
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1)

        # Show video
        cv2.imshow("Obstacle Detector", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
