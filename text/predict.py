# text/predict.py
from transformers import pipeline

# Load once (important)
classifier = pipeline("sentiment-analysis")

def get_text_emotion(text):
    if not text.strip():
        # Fallback if no text is provided
        return {"happy": 0.0, "sad": 0.0, "neutral": 1.0}

    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]

    if label == "POSITIVE":
        return {
            "happy": score,
            "sad": 0.0,
            "neutral": 1 - score
        }
    else:
        return {
            "happy": 0.0,
            "sad": score,
            "neutral": 1 - score
        }
