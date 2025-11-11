"""
SmartGuardian Cloud - version LÉGÈRE pour Render
Détecte uniquement :
🔥 Feu
🔪 Armes (couteaux, pistolets)
🌚 Intrusion de nuit
"""

import os
import time
import threading
import cv2
import numpy as np
import requests
from collections import deque
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
from ultralytics import YOLO

# ============ CONFIG ============ #
VIDEO_SOURCE = os.environ.get("VIDEO_SOURCE", "test.mp4")  # ou une caméra RTSP
FRAME_W, FRAME_H = 640, 480
INFER_W, INFER_H = 320, 240
BROWSER_FPS = 10
RECORD_FPS = 15

YOLO_MODEL_PATH = "yolov8n.pt"  # modèle léger
LARAVEL_INCIDENT_URL = os.environ.get("LARAVEL_INCIDENT_URL", "http://127.0.0.1:8000/api/incidents")

os.makedirs("clips", exist_ok=True)

# ============ GLOBALS ============ #
app = FastAPI()
camera = cv2.VideoCapture(VIDEO_SOURCE)
frame_lock = threading.Lock()
latest_annotated = None
latest_raw = None

prebuffer = deque(maxlen=60)

# Charger YOLO une seule fois
yolo = YOLO(YOLO_MODEL_PATH)

# ============ UTILITAIRES ============ #
def draw_overlay(frame, text, color=(0, 0, 255)):
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (300, 60), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.5, frame, 0.5, 0, frame)
    cv2.putText(frame, text, (20, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    return frame

def send_incident_to_laravel(type_incident, info):
    payload = {
        "camera": {"id": 1, "name": "Cam1", "location": "Zone A"},
        "incident": {
            "type_incident": type_incident,
            "date_heure": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "niveau_gravite": "élevé",
            "clip_path": "",
            "details": info,
        },
    }
    try:
        r = requests.post(LARAVEL_INCIDENT_URL, json=payload, timeout=5)
        print(f"[Laravel] Incident {type_incident} envoyé → {r.status_code}")
    except Exception as e:
        print("[Laravel] Erreur d’envoi:", e)

# ============ DÉTECTION PRINCIPALE ============ #
def detector_loop():
    global latest_annotated, latest_raw
    back_sub = cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=50, detectShadows=True)
    prev_gray = None

    while True:
        with frame_lock:
            frame = latest_raw.copy() if latest_raw is not None else None
        if frame is None:
            time.sleep(0.05)
            continue

        resized = cv2.resize(frame, (INFER_W, INFER_H))
        results = yolo(resized, conf=0.4)
        r = results[0]

        detected_labels = []
        scale_x, scale_y = FRAME_W / INFER_W, FRAME_H / INFER_H

        for box in r.boxes:
            xyxy = box.xyxy[0]
            cls = int(box.cls[0])
            label = r.names[cls]
            conf = float(box.conf[0])

            x1, y1, x2, y2 = int(xyxy[0]*scale_x), int(xyxy[1]*scale_y), int(xyxy[2]*scale_x), int(xyxy[3]*scale_y)
            detected_labels.append(label)

            if label in ["knife", "gun"]:
                frame = draw_overlay(frame, f"🔪 Arme détectée ({label})", (0, 0, 255))
                send_incident_to_laravel("arme", {"label": label, "conf": conf})

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        # Détection de feu
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_fire = np.array([0, 120, 200])
        upper_fire = np.array([35, 255, 255])
        fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
        fire_ratio = np.count_nonzero(fire_mask) / (FRAME_W * FRAME_H)
        if fire_ratio > 0.002:
            frame = draw_overlay(frame, "🔥 Feu détecté", (0, 0, 255))
            send_incident_to_laravel("feu", {"ratio": fire_ratio})

        # Intrusion dans l’obscurité
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray)
        fgmask = back_sub.apply(frame)
        moving_ratio = np.count_nonzero(fgmask) / (FRAME_W * FRAME_H)
        if moving_ratio > 0.015 and brightness < 60:
            frame = draw_overlay(frame, "🌚 Intrusion nocturne", (255, 0, 0))
            send_incident_to_laravel("intrusion", {"mouvement": moving_ratio, "luminosite": brightness})

        with frame_lock:
            latest_annotated = frame.copy()
        time.sleep(0.05)

# ============ LECTURE CAMERA ============ #
def camera_reader_thread():
    global latest_raw
    while True:
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.1)
            continue
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        with frame_lock:
            latest_raw = frame.copy()
        prebuffer.append(frame)
        time.sleep(1 / RECORD_FPS)

# ============ STREAMING ============ #
def generate_frames():
    while True:
        with frame_lock:
            frame = latest_annotated if latest_annotated is not None else latest_raw
        if frame is None:
            time.sleep(0.05)
            continue
        _, buf = cv2.imencode(".jpg", frame)
        yield (b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n")

# ============ ROUTES ============ #
@app.get("/")
def home():
    return {"message": "SmartGuardian Light — Feu, Armes, Intrusion"}

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/view", response_class=HTMLResponse)
def view_page():
    return """
    <html><head><title>SmartGuardian Light</title></head>
    <body style='background:#111;color:white;text-align:center'>
    <h2>SmartGuardian Light — Feu / Armes / Intrusion</h2>
    <img src='/video' style='width:80%;border-radius:10px'>
    </body></html>
    """

# ============ DÉMARRAGE THREADS ============ #
threading.Thread(target=camera_reader_thread, daemon=True).start()
threading.Thread(target=detector_loop, daemon=True).start()

# ============ LANCEMENT ============ #
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main_light_render:app", host="0.0.0.0", port=port)
