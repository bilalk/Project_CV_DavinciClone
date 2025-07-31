# Video Report 2: Fast and Furious Action Scene

**Date:** 2025-07-31  
**VM IP:** 40.76.115.243  
**Branch:** GPU_Testing  
**Filename:** Video_2.mp4

## ClaimsFile_2025-07-31_VMIP: 40.76.115.243 (Fast & Furious Scene)

1- Scene Detection and Change (SceneDetect - ContentDetector):  
   Original Video: Scene transition detected at 00:03:05.  
   Output Video: Scene transition detected at 00:01:22.

2- Zoom and Pan Reset at Scene Change:  
   Original Video: Zoom and pan reset occurred at 00:03:10.  
   Output Video: Zoom and pan reset occurred at 00:01:24.

3- Low-Confidence and Small-Area Filtering:  
   Original Video: Low-confidence detections filtered out at 00:02:50.  
   Output Video: Low-confidence detections filtered out at 00:01:10.

4- Subject Tracking with YOLOv8 & Motion Smoothing:  
   Output Video: Smooth tracking observed from 00:00:05 to 00:02:00.

5- Fast-Moving Object Zoom Behavior:  
   Output Video: Dynamic zoom triggered at 00:00:45, 00:01:15.

6- Scene Detection in High-Speed Motion Context:  
   Output Video: Scene cuts detected despite motion blur at 00:01:22.

7- Max Zoom Clamped at 3x:  
   Output Video: Zoom hit max limit at 00:01:05, reset at scene transition.

8- Processing Test Run on CPU (Windows 11, Local):  
   Output Video generated on local CPU without frame lag.

9- Frame-Level Stability During Fast Motion:  
   Output Video: Bounding box remained centered during high-speed sequences.

10- Zoom Responsiveness and Reset Accuracy:  
   Output Video: Zoom instantly reset on scene cut at 00:01:22.


## Notes:
- Fast-paced motion added complexity to object tracking.
- Zoom frequently adjusted based on car velocity and position.
- Scene change detector accurately captured car-to-interior transitions.
- GPU testing validated real-time performance up to 30fps at 1080p.
