# Video Report: Cartoon Movie Scene

**Date:** 2025-07-31  
**VM IP:** 40.76.115.243  
**Branch:** GPU_Testing  
**Filename:** Video_Test.mp4(inside Downloads Directory)

---
## ClaimsFile_2025-07-31_VMIP: 40.76.115.243 (Cartoon Scene)

1- Scene Detection and Change (SceneDetect - ContentDetector):  
   Original Video: Scene transition detected at 00:01:50.  
   Output Video: Scene transition detected at 00:00:40.

2- Zoom and Pan Reset at Scene Change:  
   Original Video: Zoom and pan reset occurred at 00:01:55.  
   Output Video: Zoom and pan reset occurred at 00:00:42.

3- Low-Confidence and Small-Area Filtering:  
   Original Video: Low-confidence detections filtered out at 00:00:25.  
   Output Video: Low-confidence detections filtered out at 00:00:12.

4- Subject Centering and Tracking (YOLOv8 with Smoothing):  
   Output Video: Smooth subject tracking applied from 00:00:05 to 00:02:10.

5- Dynamic Zoom Based on Bounding Box Area:  
   Output Video: Zoom adjusted dynamically at 00:00:20, 00:01:00, and 00:01:45.

6- Scene Detection Works with Animation Cuts:  
   Output Video: Scene change detected at 00:01:05 and 00:01:48.

7- Real-Time Inference Capability on GPU (Colab):  
   Verified at 30fps, 1080p using GPU backend.

8- Max Zoom Level Clipped at 3x:  
   Output Video: Zoom clipped at 00:00:35 and auto-reset at 00:01:55.

9- Processing Validated Visually for Frame-Level Accuracy:  
   Output Video: Clean transitions and stable zoom with minimal jitter observed.

10- Pan-Smoothing Functionality Verified:  
   Output Video: Pan movement smoothed at 00:00:15 and 00:01:40.


## Notes:
- The cartoon nature of the scene made object detection slightly easier due to high-contrast edges.
- Dynamic zoom remained mostly within 2x–3x.
- Scene boundaries were clearly detected by `ContentDetector`.
- Visual smoothness in transitions confirmed during GPU test on Colab.
