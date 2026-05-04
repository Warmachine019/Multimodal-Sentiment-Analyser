# app.py
from flask import Flask, render_template, request, jsonify, Response
from face.predict import get_face_data
from text.predict import get_text_emotion
from speech.predict import get_speech_data
from fusion.late_fusion import fuse
import webbrowser
from threading import Timer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    def gen():
        while True:
            _, frame = get_face_data()
            if frame:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/analyze', methods=['POST'])
def analyze():
    user_text = request.json.get("text", "")

    face_pred, _ = get_face_data()
    text_pred = get_text_emotion(user_text)
    speech_pred, speech_transcript, speech_status = get_speech_data()

    fused_scores, final_emotion, w_face, w_text, w_speech, score = fuse(face_pred, text_pred, speech_pred)

    return jsonify({
        "final_emotion": final_emotion,
        "happiness_score": round(score),
        "speech_transcript": speech_transcript,
        "speech_status": speech_status,
        "weights": {
            "face": round(w_face, 2),
            "text": round(w_text, 2),
            "speech": round(w_speech, 2)
        }
    })

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

if __name__ == '__main__':
    Timer(1, open_browser).start()
    app.run(debug=True, port=5000, use_reloader=False)
