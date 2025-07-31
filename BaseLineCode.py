# 31/7/2025 (VM_IP : 40.76.115.243) 
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import os
import time

# ========== CONFIG ==========
VIDEO_PATH = "/content/Test_200MBVid_Original.mp4"
OUTPUT_PATH = "test_output.mp4"

# ========== PARAMETERS ==========
ZOOM_MIN = 2.2
ZOOM_MAX = 3.4
CONFIDENCE_THRESHOLD = 0.35
SCENE_THRESHOLD = 30.0
TARGET_CLASS = 0  # person

# ========== INIT ==========
model = YOLO("/content/yolo12x.pt")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

deepsort = DeepSort(max_age=30)

cap = cv2.VideoCapture(VIDEO_PATH)
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
original_fps = cap.get(cv2.CAP_PROP_FPS)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")

writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, original_fps, (width, height))

# ========== SCENE DETECTION ==========
video_manager = VideoManager([VIDEO_PATH])
scene_manager = SceneManager()
scene_manager.add_detector(ContentDetector(threshold=SCENE_THRESHOLD))

video_manager.set_duration()
video_manager.start()
scene_manager.detect_scenes(frame_source=video_manager)
scene_list = scene_manager.get_scene_list()

scene_frames = [(int(start.get_frames()), int(end.get_frames())) for start, end in scene_list]
video_manager.release()

# ========== FUNCTIONS ==========

def crop_zoom(frame, center_x, center_y, zoom):
    h, w = frame.shape[:2]
    new_w = int(w / zoom)
    new_h = int(h / zoom)
    x1 = max(0, center_x - new_w // 2)
    y1 = max(0, center_y - new_h // 2)
    x2 = min(w, x1 + new_w)
    y2 = min(h, y1 + new_h)

        # Ensure cropped region is valid
    if x2 - x1 <= 0 or y2 - y1 <= 0:
        return frame  # Fallback to full frame if crop is invalid
    
    cropped = frame[y1:y2, x1:x2]
    return cv2.resize(cropped, (w, h))

# ========== PROCESS ==========
frame_idx = 0
start_time = time.time()

print("Processing started...\n")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    scene_fps = original_fps
    for s_start, s_end in scene_frames:
        if s_start <= frame_idx <= s_end:
            scene_duration = (s_end - s_start) / original_fps
            if scene_duration > 5:
                scene_fps = min(original_fps, 30)
            else:
                scene_fps = original_fps
            break

    results = model.predict(frame, conf=CONFIDENCE_THRESHOLD, device=device, verbose=False)
    detections = results[0].boxes.data.cpu().numpy()

    track_inputs = []
    for *xyxy, conf, cls in detections:
        if int(cls) == TARGET_CLASS:
            x1, y1, x2, y2 = map(int, xyxy)
            track_inputs.append(([x1, y1, x2 - x1, y2 - y1], conf, 'person'))

    tracks = deepsort.update_tracks(track_inputs, frame=frame)

    subjects = []
    for track in tracks:
        if not track.is_confirmed():
            continue
        track_box = track.to_ltrb()
        x1, y1, x2, y2 = map(int, track_box)
        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2
        box_area = (x2 - x1) * (y2 - y1)
        if box_area == 0:
            continue  # Skip invalid box
        size_ratio = box_area / (width * height)
        if size_ratio == 0:
            continue  # Extra protection
        zoom = np.clip(3.0 / size_ratio, ZOOM_MIN, ZOOM_MAX)

        subjects.append((center_x, center_y, zoom))

    if subjects:
        avg_center_x = int(np.mean([s[0] for s in subjects]))
        avg_center_y = int(np.mean([s[1] for s in subjects]))
        avg_zoom = np.mean([s[2] for s in subjects])
        frame = crop_zoom(frame, avg_center_x, avg_center_y, avg_zoom)

    writer.write(frame)
    frame_idx += 1

    # === Progress logging ===
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    elapsed = time.time() - start_time
    pct = (frame_idx / total_frames) * 100
    fps = frame_idx / elapsed if elapsed > 0 else 0
    eta = (total_frames - frame_idx) / fps if fps > 0 else 0
    print(f"[{frame_idx}/{total_frames}] {pct:.1f}% - Tempo: {elapsed:.1f}s - ETA: {eta:.1f}s")

# ========== FINALIZE ==========

writer.release()
cap.release()
print(f"\nConcluído em {time.time() - start_time:.2f} segundos.")
