# Video Report: Fast and Furious Action Scene

**Date:** 2025-07-31  
**VM IP:** 40.76.115.243  
**Branch:** GPU_Testing  
**Filename:** Video_2.mp4

---

## Summary of Implemented Features (Time-Stamped)

| TimeStamp | Feature Implemented                                     |
|-----------|---------------------------------------------------------|
| 00:00     | YOLOv8 tracking initialized with fast vehicle detection |
| 00:01     | Scene change detected due to lighting/speed shifts      |
| 00:02     | Auto-reset of zoom to full frame                        |
| 00:04     | Bounding box smoothing handled intense motion blur      |
| 00:06     | Centering shifted to nearest high-confidence target     |
| 00:08     | Motion vectors used to drive panning during car chase   |
| 00:10     | Zoom scale constrained dynamically                      |
| 00:12     | Scene change to indoor detected; zoom reset triggered   |

---

## Notes:
- Fast-paced motion added complexity to object tracking.
- Zoom frequently adjusted based on car velocity and position.
- Scene change detector accurately captured car-to-interior transitions.
- GPU testing validated real-time performance up to 30fps at 1080p.
