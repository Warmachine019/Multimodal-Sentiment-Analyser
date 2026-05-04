# fusion/late_fusion.py

def fuse(face, text, speech):
    face_conf = max(face.values())
    text_conf = max(text.values())
    speech_conf = max(speech.values())

    total = face_conf + text_conf + speech_conf

    w_face = face_conf / total
    w_text = text_conf / total
    w_speech = speech_conf / total

    final = {}
    for emotion in ["happy", "sad", "neutral"]:
        final[emotion] = (
            w_face * face[emotion] +
            w_text * text[emotion] +
            w_speech * speech[emotion]
        )

    final_emotion = max(final, key=final.get)
    
    # Calculate 0-100 Happiness Score
    happiness_score = (final["happy"] * 100) + (final["neutral"] * 50) + (final["sad"] * 0)

    return final, final_emotion, w_face, w_text, w_speech, happiness_score
