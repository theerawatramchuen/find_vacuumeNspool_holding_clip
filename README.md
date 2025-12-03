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
This system detects specific object interactions in video footage using YOLO object detection. It monitors for three key conditions:
1. __Vacuum Cleaner overlapping/inside Normal People__ (`vacuume_normal`)
2. __Vacuum Cleaner overlapping/inside Suspected People__ (`vacuume_suspected`)
3. __Spool overlapping/inside Grove__ (`spool_grove`)
   
When conditions persist for a configurable duration, the system saves:
* Original frame (no overlay)
* Annotated frame (with bounding boxes)
* 120-second video clip centered on the detection
## Key Features
* __Real-time Processing__: Processes videos at maximum speed
* __Temporal Validation__: Requires conditions to persist for a configurable time
* __Smart Restart__: Resumes detection after saved clips to avoid duplicates
* __Multi-video Support__: Processes all MP4 files in a folder sequentially
* __Visual Feedback__: Live display with processing metrics
## 👥 User Guide
## Installation
### Prerequisites
* Python 3.8+
* CUDA-compatible GPU (recommended for faster processing)
### Step-by-Step Setup
1. Clone/Download the code
2. Install dependencies:
