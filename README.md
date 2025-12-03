# Video Detection System Documentation
## find_vacuumeNspool_holding_clip
## 📋 Table of Contents
1. System Overview
2. User Guide
* Installation
* Configuration
* Usage
* Output Files
3. Developer Guide
* Architecture
* Key Classes
* Detection Logic
* Extension Guide
4. Troubleshooting
## 🎯 System Overview
## Purpose
This system detects specific object interactions in video footage using YOLOv8 object detection. It monitors for three key conditions:
1. Vacuum overlapping/inside Normal (vacuume_normal)
2. Vacuum overlapping/inside Suspected (vacuume_suspected)
3. Spool overlapping/inside Grove (spool_grove)
When conditions persist for a configurable duration, the system saves:
* Original frame (no overlay)
* Annotated frame (with bounding boxes)
* 120-second video clip centered on the detection
## Key Features
* Real-time Processing: Processes videos at maximum speed
* Temporal Validation: Requires conditions to persist for a configurable time
* Smart Restart: Resumes detection after saved clips to avoid duplicates
* Multi-video Support: Processes all MP4 files in a folder sequentially
* Visual Feedback: Live display with processing metrics

