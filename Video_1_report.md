# Video Report: Cartoon Movie Scene

**Date:** 2025-07-31  
**VM IP:** 40.76.115.243  
**Branch:** GPU_Testing  
**Filename:** Video_Test.mp4(inside Downloads Directory)

---

## Summary of Implemented Features (Time-Stamped)

| TimeStamp | Feature Implemented                                     |
|-----------|---------------------------------------------------------|
| 00:00     | YOLOv8 object detection and center tracking activated   |
| 00:01     | Bounding box smoothing and interpolation                |
| 00:03     | Motion detection used to guide zoom center              |
| 00:05     | Scene change detection triggered                        |
| 00:06     | Automatic zoom reset after scene switch                 |
| 00:08     | Subject recentered post scene-change                    |
| 00:10     | Verified zoom max level capped at 3x                    |
| 00:12     | Smooth panning applied with gradual easing              |

---

## Notes:
- The cartoon nature of the scene made object detection slightly easier due to high-contrast edges.
- Dynamic zoom remained mostly within 2x–3x.
- Scene boundaries were clearly detected by `ContentDetector`.
- Visual smoothness in transitions confirmed during GPU test on Colab.
