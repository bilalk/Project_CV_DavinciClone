import cv2
import numpy as np
import os
import time
import torch
from ultralytics import YOLO
from queue import Queue
from threading import Thread
from concurrent.futures import ThreadPoolExecutor

# ========== CONFIG ==========
VIDEO_PATH = r"C:\\Users\\pedro\\Desktop\\Scripts Filmes\\test.mp4"
OUTPUT_PATH = r"test_output.mp4"
ZOOM_MIN = 2.2
ZOOM_MAX = 3.2
SMOOTH_FACTOR = 0.01
MAX_MOVE = 30
MAX_ZOOM_DELTA = 0.01
PRINT_EVERY_N_FRAMES = 50
BATCH_SIZE = 1
FRAME_QUEUE_SIZE = 1

# ========== DEVICE ==========
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("CUDA disponível" if device.type == "cuda" else "CUDA não disponível, rodando com CPU")

# ========== MODELO ==========
model = YOLO("yolo12X.pt")
model.to(device)
print("Modelo YOLO 12X carregado")

# ========== VÍDEO ==========
cap = cv2.VideoCapture(VIDEO_PATH)
if not cap.isOpened():
    raise IOError("Erro ao abrir o vídeo.")

fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
w_original = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h_original = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, fps, (w_original, h_original))

# ========== VARIÁVEIS ==========
x_center_suave = w_original // 2
y_center_suave = h_original // 2
escala_suave = ZOOM_MIN
start_time = time.time()
frame_count = 0

# ========== LEITURA MULTITHREAD ==========
frame_queue = Queue(maxsize=FRAME_QUEUE_SIZE)

def multi_reader(video_path, queue, thread_id):
    cap_local = cv2.VideoCapture(video_path)
    frame_id = 0
    while cap_local.isOpened():
        ret, frame = cap_local.read()
        if not ret:
            break
        if frame_id % 2 == thread_id:
            queue.put(frame)
        frame_id += 1
    cap_local.release()
    queue.put(None)

reader1 = Thread(target=multi_reader, args=(VIDEO_PATH, frame_queue, 0))
reader2 = Thread(target=multi_reader, args=(VIDEO_PATH, frame_queue, 1))
reader1.start()
reader2.start()

# ========== LOOP PRINCIPAL ==========
active_readers = 1
batch = []

while active_readers > 0 or not frame_queue.empty():
    while len(batch) < BATCH_SIZE and not frame_queue.empty():
        frame = frame_queue.get()
        if frame is None:
            active_readers -= 1
        else:
            batch.append(frame)

    if not batch:
        continue

    # PRÉ-PROCESSAMENTO
    batch_rgb = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in batch]

    # INFERÊNCIA
    results_list = model(batch_rgb, verbose=False, device=device)

    for i, results in enumerate(results_list):
        frame = batch[i]
        best_boxes = []

        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            score = float(box.conf[0])
            class_id = int(box.cls[0])

            w = x2 - x1
            h = y2 - y1
            area = w * h
            cx, cy = x1 + w // 2, y1 + h // 2

            dist = np.hypot(cx - w_original / 2, cy - h_original / 2)
            centrality_score = max(0.001, 1 / (1 + dist))
            final_score = score * area * centrality_score
            best_boxes.append((final_score, (x1, y1, w, h)))

        best_boxes.sort(reverse=True)

        # CENTRO E ZOOM
        if len(best_boxes) >= 2:
            _, (x1, y1, w1, h1) = best_boxes[0]
            _, (x2, y2, w2, h2) = best_boxes[1]
            cx1, cy1 = x1 + w1 // 2, y1 + h1 // 2
            cx2, cy2 = x2 + w2 // 2, y2 + h2 // 2
            x_center = int((cx1 + cx2) / 2)
            y_center = int((cy1 + cy2) / 2)
            span = max(abs(cx1 - cx2), abs(cy1 - cy2)) + (w1 + w2) // 4
            escala = np.clip(w_original / span, ZOOM_MIN, ZOOM_MAX)

        elif len(best_boxes) == 1:
            _, (x, y, w, h) = best_boxes[0]
            x_center = x + w // 2
            y_center = y + h // 2
            escala = np.clip(w_original / (w + 1), ZOOM_MIN, ZOOM_MAX)
        else:
            x_center = w_original // 2
            y_center = h_original // 2
            escala = ZOOM_MIN

        # SUAVIZAÇÃO
        dx = x_center - x_center_suave
        dy = y_center - y_center_suave
        dist = np.hypot(dx, dy)
        if dist > MAX_MOVE:
            fator = MAX_MOVE / dist
            dx *= fator
            dy *= fator

        x_center_suave += int(dx * SMOOTH_FACTOR)
        y_center_suave += int(dy * SMOOTH_FACTOR)

        escala_diff = escala - escala_suave
        escala_diff = np.clip(escala_diff, -MAX_ZOOM_DELTA, MAX_ZOOM_DELTA)
        escala_suave += escala_diff

        # RECORTE
        new_w = int(w_original / escala_suave)
        new_h = int(h_original / escala_suave)
        x1 = max(0, min(x_center_suave - new_w // 2, w_original - new_w))
        y1 = max(0, min(y_center_suave - new_h // 2, h_original - new_h))

        crop = frame[y1:y1 + new_h, x1:x1 + new_w]
        resized = cv2.resize(crop, (w_original, h_original))
        writer.write(resized.copy())

        frame_count += 1
        if frame_count % PRINT_EVERY_N_FRAMES == 0:
            elapsed = time.time() - start_time
            pct = 100 * frame_count / total_frames
            eta = elapsed / frame_count * (total_frames - frame_count)
            print(f"[{frame_count}/{total_frames}] {pct:.1f}% - Tempo: {elapsed:.1f}s - ETA: {eta:.1f}s")

    batch.clear()

# FINAL
writer.release()
print(f"\nConcluído em {time.time() - start_time:.2f} segundos.")
