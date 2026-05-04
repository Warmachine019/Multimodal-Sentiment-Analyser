# face/predict.py
import cv2
from fer.fer import FER
import threading

detector = FER()
camera = cv2.VideoCapture(0)

latest_emotion = {"happy": 0.0, "sad": 0.0, "neutral": 1.0}
latest_frame = None

def generate_frames():
    global latest_emotion, latest_frame
    while True:
        success, frame = camera.read()
        if not success:
            continue
        
        result = detector.detect_emotions(frame)
        if result:
            emotions = result[0]["emotions"]
            box = result[0]["box"]
            x, y, w, h = box
            
            happy = emotions["happy"]
            sad = emotions["sad"] + emotions["fear"] + emotions["angry"]
            neutral = emotions["neutral"] + emotions["surprise"] + emotions["disgust"]
            
            total = happy + sad + neutral
            if total > 0:
                latest_emotion = {
                    "happy": happy / total,
                    "sad": sad / total,
                    "neutral": neutral / total
                }
            
            # Determine dominant emotion for bounding box UI
            dom_emotion = max(latest_emotion, key=latest_emotion.get)
            if dom_emotion == "happy":
                color = (0, 255, 0) # Green
                label = "Happy"
            elif dom_emotion == "sad":
                color = (0, 0, 255) # Red
                label = "Sad"
            else:
                color = (128, 128, 128) # Gray
                label = "Neutral"
            
            # Draw box and text
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
            
        ret, buffer = cv2.imencode('.jpg', frame)
        latest_frame = buffer.tobytes()

threading.Thread(target=generate_frames, daemon=True).start()

def get_face_data():
    return latest_emotion, latest_frame
