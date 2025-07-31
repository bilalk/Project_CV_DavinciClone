# Project_CV_DavinciClone or Auto-Reframing & Smart Zoom with Scene Detection (Colab + GPU)

This project performs auto-reframing of a video using:

YOLOv8 (for object/person detection)

Norfair (for object tracking)

SceneDetect (for automatic scene/shot boundary detection)

Intelligent zoom/pan logic to maintain a ~300% zoom level while keeping subjects in focus.

## Features
 Person-centric reframing using bounding boxes and centroid tracking

 Smooth pan and zoom transitions between frames

 Automatic zoom reset or adjustment based on shot changes

 Scene detection using scenedetect to adapt behavior on cut boundaries

Colab-ready setup (GPU-enabled)


## Build Instructions (For Colab GPU):

!pip install -U pip setuptools wheel cmake build \
&& pip install git+https://github.com/rlabbe/filterpy.git \
&& pip install git+https://github.com/tryolabs/norfair.git --no-build-isolation \
&& pip install ultralytics scenedetect ffmpeg-python opencv-python


## How it Works

Detection & Tracking:
YOLOv8 detects people in each frame.
Norfair tracks these people across frames by associating detections with IDs.

Smart Zoom & Reframing:

Computes a zoom level (~3x) dynamically.
Determines a central point using average subject position.
Crops and resizes the frame around that subject intelligently.

Scene Change Handling:

scenedetect finds shot boundaries.
Zoom and pan smoothly reset or transition when scene/shot changes are detected.

##  How to Run:

Upload your video as input.mp4 in Colab.

Run main.ipynb.

Final video will be saved as test_output.mp4.
