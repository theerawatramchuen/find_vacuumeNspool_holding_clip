# Video Detection System Documentation
## find_vacuumeNspool_holding_clip
<img width="638" height="664" alt="image" src="https://github.com/user-attachments/assets/6b837ced-6801-4d1b-959a-4fb26f98c84b" />

## python video_reviewer.py
<img width="1365" height="767" alt="image" src="https://github.com/user-attachments/assets/fea4581e-31c3-459a-ac96-87165e96526b" />

## python video_reviewer_2.py
<img width="1346" height="1031" alt="image" src="https://github.com/user-attachments/assets/d3983fdb-b29a-4aa0-9b65-d2b5faa7b01d" />


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
```
bash
pip install ultralytics opencv-python pandas numpy
```
3. Download YOLO model:
* Place your trained model at: `C:\Users\RYZEN\cctv_mon\project_yolov11_obb\runs\detect\train3\weights\best.pt`
* Or modify MODEL_PATH in the main section
* Or unzip best.7z001 - 009 in weight folder to get "best.pt"
## Configuration
Edit these parameters in the main section (lines 308-313):
```
python
MODEL_PATH = r"C:\Users\RYZEN\cctv_mon\project_yolov11_obb\runs\detect\train3\weights\best.pt"
VIDEO_FOLDER = r"C:\RecordDownload"  # Folder containing .mp4 videos
CONFIDENCE_THRESHOLD = 0.5  # 0.0 to 1.0 (higher = more confident detections)
LINE_THICKNESS = 1  # Bounding box thickness
VALIDATION_TIME = 2.0  # Seconds condition must persist
```
## Usage
### Running the System
```
bash
python find_vacuumeNspool_holding.py
```
### Controls During Execution
* `q`: Quit processing
* `Space`: Pause/resume video
* __Window Close__: Exit program
### Processing Flow
1. System loads all .mp4 files from VIDEO_FOLDER
2. Processes each video sequentially
3. Displays real-time detection with overlay
4. Saves detections when conditions meet validation criteria
5. Automatically moves to next video when complete
## Output Files
### Directory Structure
```
text
vacuume-spool/
├── 20231201-143045-12-O.jpg      # Original frame (no overlay)
├── 20231201-143045-12-R.jpg      # Result frame (with bounding boxes)
├── 20231201-143045-12.mp4        # 120-second video clip
└── ...
```
### File Naming Convention
```
text
YYYYMMDD-HHMMSS-milliseconds-type.ext
```
* __O__: Original frame
* __R__: Result frame (with annotations)
* __.mp4__: Video clip (no annotations)
### Video Clip Details
* Duration: 120 seconds total
* Positioning: Detection at center (60s before, 60s after)
* Content: Original video frames, no overlays
* Format: MP4 with same encoding as source
## 👨‍💻 Developer Guide
### Architecture
```
text
DetailedVideoYOLOInference
├── __init__()                    # Initialize model and parameters
├── process_video_file()          # Main video processing loop
├── process_detections()          # Core detection logic
├── save_video_clip()             # Save 120-second clips
├── is_inside_or_overlapping()    # Spatial relationship check
└── calculate_iou()              # Intersection-over-Union calculation
```
### Key Classes
`DetailedVideoYOLOInference`
Main controller class handling:
* Model loading and inference
* Video processing pipeline
* Condition tracking and validation
* File output management

### Key Attributes:
* `condition_frames`: Tracks condition persistence across frames
* `validated_conditions`: Set of validated conditions
* `model`: YOLO model instance
* `running`: Control flag for graceful shutdown
### Detection Logic
1. __Frame Processing Pipeline__
```
text
Read Frame → YOLO Inference → Class Filtering → Spatial Analysis → Condition Tracking → Output Trigger
```
2. __Condition Tracking__
* Each condition gets a unique key based on object positions
* Tracks persistence across consecutive frames
* Validates after `validation_time * fps` frames
* Resets tracking when condition disappears

3. __Spatial Analysis Methods__
* `calculate_iou()`: Computes overlap ratio between boxes
* `is_inside_or_overlapping()`: Checks if one box is inside/overlapping another
* `get_condition_key()`: Creates unique identifier for condition tracking
## Extension Guide
### Adding New Detection Conditions
1. __Modify__ `process_detections()` __method:__
```
python
# Example: Add "tool in/overlapping machine"
if 'tool' in detections_by_class and 'machine' in detections_by_class:
    for tool_det in detections_by_class['tool']:
        for machine_det in detections_by_class['machine']:
            if self.is_inside_or_overlapping(tool_det['bbox'], machine_det['bbox']):
                condition_key = self.get_condition_key("tool_machine", tool_det, machine_det)
                current_conditions.add(condition_key)
                # ... rest of tracking logic
```
2. __Update condition logging__ (in the same method):
```
python
elif "tool_machine" in condition_key:
    overlap_info.append(f"tool in/overlapping machine (validated for {frames_elapsed} frames)")
```
### Modifying Output Behavior
### Change clip duration:
```
python
# In __init__ method
self.clip_target_duration = 180  # 3 minutes total
self.clip_before_duration = 90   # 1.5 minutes before
self.clip_after_duration = 90    # 1.5 minutes after
```
### Modify output directory:
```
python
# In __init__ method
self.save_dir = "custom-output-folder"
```
### Adding New Output Formats
Extend the saving logic in `process_detections()`:
```
python
# Example: Save JSON metadata
import json
metadata = {
    "timestamp": datetime.now().isoformat(),
    "frame_number": current_frame_number,
    "condition": condition_key,
    "objects": object_details
}
metadata_filename = f"{filename_base}.json"
with open(os.path.join(self.save_dir, metadata_filename), 'w') as f:
    json.dump(metadata, f, indent=2)
```
## Performance Optimization
### For Faster Processing:
1. __Reduce input resolution__ in YOLO model
2. __Lower confidence threshold__ to reduce false positives
3. __Use batch processing__ for multiple frames (modify inference call)
4. __Implement frame skipping__ for high-FPS videos
### Memory Management:
* Frames are processed and released immediately
* Only clip frames are buffered temporarily
* Model runs on GPU if available (automatic with ultralytics)
