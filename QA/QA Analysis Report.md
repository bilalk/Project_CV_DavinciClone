# QA Analysis Report: Enhanced Video Processing Pipeline

## Overview

This repository contains an enhanced video processing pipeline combining:

- Object Detection with YOLOv8x
- Multi-object Tracking using DeepSORT
- Scene Detection with PySceneDetect
- Smart reframing using zoom & pan smoothing
- Post-processing video formatting using FFmpeg

This report documents key parameters, hardware info, and output metrics for QA testing, tuning, and reproducibility.

---

## Algorithmic Parameters to Test and Report

### A. Object Detection (YOLO)

- **Model Type/Version:** YOLOv8x (`yolov8x.pt`)
- **Detection Confidence Threshold:** Default 0.35 (configurable)
- **Classes to Track:** Dynamic per scene, default `["person", "car"]`
- **NMS (Non-Maximum Suppression) Threshold:** 0.5
- **Input Image Size:** 640x640 (resized input)
- **Batch Size for Inference:** 4 (configurable)

### B. Object Tracking (DeepSORT)

- **Max Age:** 30 frames
- **n_init:** 3 frames (min frames before confirming a track)
- **Track Confidence Threshold:** Handled internally by DeepSORT
- **IoU Matching Threshold:** 0.7
- **Feature Embedding Model/Distance Threshold:** DeepSORT defaults

### C. Scene Detection (PySceneDetect)

- **Scene Change Threshold:** 30.0
- **Detector Type:** ContentDetector
- **Minimum Scene Duration:** Adjustable, implicit default

### D. Reframe / Camera Logic

- **Zoom Min / Max:** 2.2x to 3.4x
- **Target Zoom:** Adaptive ~3.0 based on subject size
- **Pan / Zoom Smoothing Factor:** EMA with factor 0.2
- **Max Camera Move Per Frame:** Smoothed by EMA, no explicit cap
- **Aspect Ratio Output:** Original input (e.g., 3840x2160)
- **Subject Selection:** Average center & zoom of all confirmed subjects
- **Fallback Policy:** No pan/zoom if no subject detected

### E. GPU / Hardware

- **GPU Model:** RunPod A4000
- **Inference Precision:** PyTorch default (fp32)
- **Batch Size:** 4 for YOLO inference
- **Video Input Resolution:** Dynamically detected
- **Video Output Resolution:** Matches input
- **FPS:** Configurable output FPS (default 24); actual processing FPS reported

### F. Output / Reporting Metrics

- Scene segmentation accuracy (scene counts and frame ranges)
- Tracking jitter (pixels/frame)
- Zoom level statistics (mean/std)
- Pan speed statistics (derived from center movement)
- Total scenes processed
- Total processing time (seconds)
- Subject coverage rate (frames with confirmed subjects vs total frames)

---

## How to Run

**Setup environment**

```bash
pip install ultralytics deep_sort_realtime scenedetect opencv-python-headless GPUtil psutil numpy
```

### 1\. Configure

Update the `CONFIG` dictionary with:

-   `video_path`: Input video file path

-   `output_path` and `final_output_path`: Output video paths

-   Thresholds and classes as needed

### 2\. Purpose

-   Tune detection, tracking, and smoothing parameters for best reframing

-   Ensure reproducibility across hardware and video types

-   Debug jitter, missed detections, and scene detection errors

-   Scale pipeline for different video genres and scenarios
