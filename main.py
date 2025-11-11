from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
from ultralytics import YOLO
import numpy as np

app = FastAPI(title="SmartGuardian Light")

# Charger ton modèle YOLO
model = YOLO("best.pt")  # Remplace par ton modèle

# Ouvrir la webcam locale
cap = cv2.VideoCapture(0)

def generate_frames():
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Faire la prédiction avec YOLO
        results = model(frame)

        # Dessiner les boîtes et labels sur l'image
        for box in results[0].boxes:
            cls_id = int(box.cls)
            conf = float(box.conf)
            name = model.names[cls_id]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"{name} {conf:.2f}", (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Encoder l'image en JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        # Générer la frame pour le streaming
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.get("/view")
def video_feed():
    """Page HTML pour afficher la caméra"""
    html_content = """
    <html>
        <head>
            <title>SmartGuardian Light</title>
        </head>
        <body>
            <h1>SmartGuardian Light — Feu, Armes, Intrusion</h1>
            <img src="/video" width="800" />
        </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.get("/video")
def video_stream():
    """Flux vidéo MJPEG"""
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

# Fermer la caméra proprement à l'arrêt
import atexit
atexit.register(lambda: cap.release())
