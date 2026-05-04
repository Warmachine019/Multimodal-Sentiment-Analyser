# speech/predict.py
import speech_recognition as sr
from text.predict import get_text_emotion
import threading
import time

latest_speech_emotion = {"happy": 0.0, "sad": 0.0, "neutral": 1.0}
current_transcript = "Speak into your microphone..."
speech_status = "Listening..."

def continuous_speech_to_text():
    global latest_speech_emotion, current_transcript, speech_status
    recognizer = sr.Recognizer()
    mic = sr.Microphone()
    
    # Calibrate background noise once on startup
    with mic as source:
        recognizer.adjust_for_ambient_noise(source, duration=1)
        
    while True:
        try:
            with mic as source:
                speech_status = "Listening..."
                # Listens until it detects a pause (silence)
                audio = recognizer.listen(source, phrase_time_limit=10)
                
            speech_status = "Processing..."
            time.sleep(0.5) # Give the UI a fraction of a second to show "Processing..."
            
            # Convert speech to text using Google's free API
            text = recognizer.recognize_google(audio)
            current_transcript = text
            
            # RE-USE the text sentiment model on the spoken text!
            latest_speech_emotion = get_text_emotion(text)
            speech_status = "Done"
            time.sleep(1) # Hold "Done" status for a second before listening again
            
        except sr.UnknownValueError:
            speech_status = "Listening..." # Kept quiet, just go back to listening
        except sr.RequestError:
            speech_status = "API Offline"
            time.sleep(2)
        except Exception as e:
            time.sleep(1)

threading.Thread(target=continuous_speech_to_text, daemon=True).start()

def get_speech_data():
    return latest_speech_emotion, current_transcript, speech_status
