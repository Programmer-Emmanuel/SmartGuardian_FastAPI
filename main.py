# main_no_mediapipe.py
"""
SmartGuardian Cloud - multi-detectors (version WITHOUT MediaPipe)
Detecte: weapon (knife/gun), fight (via action model placeholder), fall (heuristique bbox),
crowd, accident (optical flow + vehicle), fire (color heuristic), intrusion (background sub),
vandalism (sudden disappearance).
Enregistre un clip de CLIP_DURATION secondes (pré-buffer + post) et poste un incident vers Laravel.
"""

import os
import time
import threading
import queue
from collections import deque
import json
from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
import requests

# ML libs
from ultralytics import YOLO
import torch
import torchvision.transforms as T
from torchvision.models.video import r3d_18

# ============ CONFIG ============
CAM_IDX = 0
FRAME_W, FRAME_H = 640, 480
INFER_W, INFER_H = 320, 240
BROWSER_FPS = 12
RECORD_FPS = 20
CLIP_DURATION = 10
PREBUFFER_SECONDS = 3
PREBUFFER_FRAMES = PREBUFFER_SECONDS * RECORD_FPS

YOLO_MODEL_PATH = "yolov8s.pt"
ACTION_MODEL_DEVICE = "cpu"
ACTION_CLIP_LEN = 16
ACTION_THRESHOLD = 0.6
LARAVEL_INCIDENT_URL = "https://smart-guradian.onrender.com//api/incidents"

os.makedirs("clips", exist_ok=True)

# ============ Globals & queues ============
app = FastAPI()
camera = cv2.VideoCapture(CAM_IDX)
camera.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_W)
camera.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_H)

frame_lock = threading.Lock()
latest_annotated = None
latest_raw = None

prebuffer = deque(maxlen=PREBUFFER_FRAMES)
recording = False
record_lock = threading.Lock()
recording_queue = deque()

action_request_q = queue.Queue(maxsize=4)
action_result_q = queue.Queue(maxsize=4)

# ============ Models init ============
yolo = YOLO(YOLO_MODEL_PATH)
device = ACTION_MODEL_DEVICE
action_model = r3d_18(pretrained=True)
action_model.eval().to(device)

action_transform = T.Compose([
    T.ToPILImage(),
    T.Resize((112, 112)),
    T.ToTensor(),
    T.Normalize(mean=[0.43216,0.394666,0.37645], std=[0.22803,0.22145,0.216989])
])

# ============ Utilities ============
def draw_overlay(frame, text_lines, color=(0,0,255)):
    h, w = frame.shape[:2]
    box_w = min(340, w - 10)
    box_h = 20 + 18 * len(text_lines)
    overlay = frame.copy()
    cv2.rectangle(overlay, (5,5), (5 + box_w, 8 + box_h), (0,0,0), -1)
    alpha = 0.45
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
    y = 25
    for line in text_lines:
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
        y += 18

def save_clip_from_buffer(frames, filename):
    if not frames:
        return None
    path = os.path.join("clips", filename)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(path, fourcc, RECORD_FPS, (FRAME_W, FRAME_H))
    for f in frames:
        out.write(f)
    out.release()
    return path

def send_incident_to_laravel(payload):
    try:
        r = requests.post(LARAVEL_INCIDENT_URL, json=payload, timeout=5)
        print("[webhook] sent, status", r.status_code)
    except Exception as e:
        print("[webhook] error:", e)

# ============ Action recognition worker ============
def frames_to_action_tensor(frames_list):
    with torch.no_grad():
        imgs = [action_transform(cv2.cvtColor(f, cv2.COLOR_BGR2RGB)) for f in frames_list]
        clip = torch.stack(imgs, dim=1)  # C x T x H x W
        clip = clip.unsqueeze(0).to(device)
        return clip

def action_worker():
    while True:
        try:
            clip_frames = action_request_q.get(timeout=1)
        except queue.Empty:
            continue
        try:
            tensor = frames_to_action_tensor(clip_frames)
            with torch.no_grad():
                out = action_model(tensor)
                probs = torch.nn.functional.softmax(out, dim=1)
                top_prob, top_idx = torch.max(probs, dim=1)
                score = float(top_prob.item())
                idx = int(top_idx.item())
                label = f"k_{idx}"
                action_result_q.put((label, score))
        except Exception as e:
            print("[action_worker] error", e)
            action_result_q.put((None, 0.0))

threading.Thread(target=action_worker, daemon=True).start()

# ============ Detector loop ============
def detector_loop():
    global latest_annotated, latest_raw, recording, recording_queue
    back_sub = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
    prev_gray = None
    last_objects = {}

    while True:
        with frame_lock:
            raw = latest_raw.copy() if latest_raw is not None else None
        if raw is None:
            time.sleep(0.02)
            continue

        small = cv2.resize(raw, (INFER_W, INFER_H))
        results = yolo(small, conf=0.35)
        r = results[0]

        scale_x = FRAME_W / INFER_W
        scale_y = FRAME_H / INFER_H
        detected_objects = []
        person_count = 0
        vehicle_count = 0
        dangerous_detected = []

        for box in r.boxes:
            try:
                xyxy = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy[0], 'cpu') else np.array(box.xyxy[0])
            except Exception:
                xyxy = np.array(box.xyxy[0])
            try:
                cls = int(box.cls[0].cpu().numpy()) if hasattr(box.cls[0], 'cpu') else int(box.cls[0])
            except Exception:
                cls = int(box.cls[0])
            try:
                conf = float(box.conf[0].cpu().numpy()) if hasattr(box.conf[0], 'cpu') else float(box.conf[0])
            except Exception:
                conf = float(box.conf[0])
            label = r.names[cls] if hasattr(r, 'names') else yolo.names[cls]
            x1,y1,x2,y2 = int(xyxy[0]*scale_x), int(xyxy[1]*scale_y), int(xyxy[2]*scale_x), int(xyxy[3]*scale_y)
            detected_objects.append((label, conf, (x1,y1,x2,y2)))
            if label == "person":
                person_count += 1
            if label in {"car","truck","bus","motorcycle"}:
                vehicle_count += 1
            if label in {"knife","gun"}:
                dangerous_detected.append((label, conf, (x1,y1,x2,y2)))

        # Fall heuristic
        fall_detected = False
        for lbl, conf, box in detected_objects:
            if lbl == "person" and conf > 0.4:
                x1,y1,x2,y2 = box
                w = max(1, x2 - x1)
                h = max(1, y2 - y1)
                aspect = h / w
                bottom_dist = FRAME_H - y2
                if aspect < 1.2 and bottom_dist < FRAME_H * 0.25:
                    fall_detected = True

        # Motion magnitude
        gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)
        motion_magnitude = 0.0
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5,3,15,3,5,1.2,0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            motion_magnitude = float(np.mean(mag))
        prev_gray = gray

        # Background subtraction intrusion
        fgmask = back_sub.apply(raw)
        moving_area = np.count_nonzero(fgmask) / (FRAME_W * FRAME_H)

        # Fire heuristic
        hsv = cv2.cvtColor(raw, cv2.COLOR_BGR2HSV)
        lower_fire = np.array([0, 120, 200])
        upper_fire = np.array([35, 255, 255])
        fire_mask = cv2.inRange(hsv, lower_fire, upper_fire)
        fire_ratio = np.count_nonzero(fire_mask) / (FRAME_W * FRAME_H)

        # Crowd density
        crowd_density = person_count / (FRAME_W * FRAME_H) * 1e6

        # Vandalism
        current_labels = [o[0] for o in detected_objects]
        vandalism_flag = False
        if motion_magnitude > 2.0 and any(x in last_objects for x in ["tv","vase","bench","chair"]):
            for pres in ["tv","vase","chair","bench"]:
                if pres in last_objects and pres not in current_labels:
                    vandalism_flag = True
        last_objects = {o[0]: o for o in detected_objects}

        # Decide incidents
        incidents = []

        for (label, conf, box) in dangerous_detected:
            if conf > 0.45:
                incidents.append(("weapon", {"label": label, "conf": conf, "box": box}))

        if fall_detected:
            incidents.append(("fall", {"score": 0.9}))

        if person_count >= 6 or crowd_density > 0.12:
            incidents.append(("crowd", {"persons": person_count, "density": crowd_density}))

        if fire_ratio > 0.001:
            incidents.append(("fire", {"ratio": float(fire_ratio)}))

        mean_brightness = float(np.mean(cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY)))
        if moving_area > 0.01 and mean_brightness < 60:
            incidents.append(("intrusion", {"moving_area": moving_area, "brightness": mean_brightness}))

        if motion_magnitude > 2.5 and vehicle_count > 0:
            incidents.append(("accident", {"motion": motion_magnitude, "vehicles": vehicle_count}))

        if vandalism_flag:
            incidents.append(("vandalism", {"motion": motion_magnitude}))

        # Action recognition
        need_action_check = any(t in ("crowd","accident","vandalism") for t,_ in incidents) or len(dangerous_detected)>0
        if need_action_check:
            with frame_lock:
                clip_frames = list(prebuffer)[-ACTION_CLIP_LEN:] if len(prebuffer) >= ACTION_CLIP_LEN else list(prebuffer)
                while len(clip_frames) < ACTION_CLIP_LEN:
                    clip_frames.insert(0, clip_frames[0] if clip_frames else (latest_raw.copy() if latest_raw is not None else np.zeros((FRAME_H,FRAME_W,3),dtype=np.uint8)))
            try:
                action_request_q.put_nowait(clip_frames)
            except queue.Full:
                pass
            try:
                label, score = action_result_q.get_nowait()
                if label == "fight" or (label.startswith("k_") and score > ACTION_THRESHOLD):
                    incidents.append(("fight", {"label": label, "score": score}))
            except queue.Empty:
                pass

        # Annotate & trigger<a
        if incidents:
            annotated = raw.copy()
            text_lines = []
            for inst, meta in incidents:
                text_lines.append(f"{inst}: {meta}")
            for (label, conf, box) in dangerous_detected:
                x1,y1,x2,y2 = box
                cv2.rectangle(annotated, (x1,y1), (x2,y2), (0,0,255), 2)
                cv2.putText(annotated, f"{label} {conf:.2f}", (x1,y1-6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
            draw_overlay(annotated, text_lines, color=(0,0,255))
            with frame_lock:
                latest_annotated = annotated.copy()

            with record_lock:
                if not recording:
                    print("[detector] incident detected -> start recording + webhook")
                    recording = True
                    with frame_lock:
                        for f in list(prebuffer):
                            recording_queue.append(f.copy())
                    threading.Thread(target=handle_incident, args=(incidents,), daemon=True).start()
        else:
            with frame_lock:
                latest_annotated = raw.copy()

        with frame_lock:
            prebuffer.append(raw.copy())

        time.sleep(0.03)

# ============ handle incident (séparé) ============
def handle_incident(incidents):
    global recording, recording_queue

    post_frames_needed = CLIP_DURATION * RECORD_FPS
    collected = []
    while len(collected) < post_frames_needed:
        with record_lock:
            if recording_queue:
                collected.append(recording_queue.popleft())
            else:
                with frame_lock:
                    if latest_annotated is not None:
                        collected.append(latest_annotated.copy())
                    elif latest_raw is not None:
                        collected.append(latest_raw.copy())
                    else:
                        collected.append(np.zeros((FRAME_H,FRAME_W,3),dtype=np.uint8))
        time.sleep(1/RECORD_FPS)

    filename = f"incident_{int(time.time())}.mp4"
    path = save_clip_from_buffer(collected, filename)
    print("[handle_incident] Clip saved:", path)

    camera_data = {"id": 1, "name": "Cam1", "location": "Hall A"}
    incident_data = {
        "type_incident": ",".join([i[0] for i in incidents]),
        "date_heure": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "niveau_gravite": "élevé",
        "clip_path": path,
        "details": incidents
    }
    payload = {"camera": camera_data, "incident": incident_data}

    print("\n[INCIDENT DETECTED]")
    print("Camera Info:", json.dumps(camera_data, indent=2))
    print("Incident Info:", json.dumps(incident_data, indent=2))

    threading.Thread(target=send_incident_to_laravel, args=(payload,), daemon=True).start()

    with record_lock:
        recording = False
        recording_queue.clear()

# ============ camera reader ============
def camera_reader_thread():
    global latest_raw, recording
    while True:
        ok, frame = camera.read()
        if not ok:
            time.sleep(0.05)
            continue
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        with frame_lock:
            latest_raw = frame.copy()
            if recording:
                with record_lock:
                    recording_queue.append(frame.copy())
            prebuffer.append(frame.copy())
        time.sleep(1.0 / RECORD_FPS)

# ============ streaming generator ============
def generate_frames():
    last_send = 0
    min_interval = 1.0 / BROWSER_FPS
    while True:
        now = time.time()
        if now - last_send < min_interval:
            time.sleep(min_interval - (now - last_send))
        last_send = time.time()

        with frame_lock:
            frame = latest_annotated.copy() if latest_annotated is not None else (latest_raw.copy() if latest_raw is not None else None)
        if frame is None:
            continue

        with record_lock:
            if recording:
                cv2.circle(frame, (FRAME_W-30,30), 10, (0,0,255), -1)
                cv2.putText(frame, "REC", (FRAME_W-70,35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

        ret, buf = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        if not ret:
            continue
        jpg = buf.tobytes()
        yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpg + b'\r\n')

# ============ FastAPI routes ============
@app.get("/")
def home():
    return {"message": "SmartGuardian Cloud - multi-detectors (no mediapipe)"}

@app.get("/video")
def video_feed():
    return StreamingResponse(generate_frames(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/view", response_class=HTMLResponse)
def view_page():
    return """
    <html><head><title>SmartGuardian Dashboard</title></head>
    <body style='margin:0;background:#111;color:#fff;font-family:Arial;display:flex;flex-direction:column;align-items:center'>
    <h2 style='margin:12px'>SmartGuardian Cloud - Dashboard (no mediapipe)</h2>
    <img src="/video" style='width:90%;max-width:1280px;border-radius:8px;box-shadow:0 10px 40px rgba(0,0,0,0.6)'/>
    </body></html>
    """

# ============ Start threads ============
threading.Thread(target=camera_reader_thread, daemon=True).start()
threading.Thread(target=detector_loop, daemon=True).start()
# action_worker déjà lancé
