import cv2
import pandas as pd
import os
from datetime import datetime
import time
import logging
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import glob

class DetailedVideoYOLOInference:
    def __init__(self, model_path, video_folder, confidence_threshold=0.5, line_thickness=2, validation_time=2.0):
        self.logger = logging.getLogger(__name__)
        self.model_path = model_path
        self.video_folder = video_folder
        self.confidence_threshold = confidence_threshold
        self.line_thickness = line_thickness
        self.validation_time = validation_time  # Validation time in seconds
        self.current_video_index = 0
        self.save_dir = "vacuume-spool"
        self.running = True
        
        # Track conditions over frames
        self.condition_frames = {}  # Store frame count for each condition
        self.validated_conditions = set()  # Track conditions that have been validated
        
        # Video clip recording variables
        self.clip_recording = False
        self.clip_frames = []
        self.clip_start_frame = 0
        self.clip_duration_frames = 0  # Will be calculated based on video FPS
        self.clip_target_duration = 120  # 120 seconds total clip
        self.clip_before_duration = 60  # 60 seconds before detection
        self.clip_after_duration = 60   # 60 seconds after detection
        
        # Initialize YOLO model
        self.model = YOLO(model_path)
        
        # Load video files from folder
        self.video_files = glob.glob(os.path.join(video_folder, "*.mp4"))
        
        if not self.video_files:
            raise ValueError(f"No .mp4 files found in folder: {video_folder}")
        
        self.logger.info(f"Loaded {len(self.video_files)} video files")
        self.logger.info(f"Model classes: {self.model.names}")
        self.logger.info(f"Confidence threshold: {self.confidence_threshold}")
        self.logger.info(f"Line thickness: {self.line_thickness}")
        self.logger.info(f"Validation time: {validation_time} seconds")
    
    def generate_filename(self, image_type):
        """Generate filename in required format"""
        now = datetime.now()
        milliseconds = int(now.microsecond / 10000)
        return f"{now.strftime('%Y%m%d-%H%M%S')}-{milliseconds}-{image_type}"
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union (IoU) between two bounding boxes"""
        # Convert boxes to [x1, y1, x2, y2] format if needed
        if hasattr(box1, 'cpu'):
            box1 = box1.cpu().numpy()
        if hasattr(box2, 'cpu'):
            box2 = box2.cpu().numpy()
            
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = box1_area + box2_area - intersection_area
        
        # Calculate IoU
        iou = intersection_area / union_area if union_area > 0 else 0
        return iou
    
    def is_inside_or_overlapping(self, inner_box, outer_box, iou_threshold=0.1):
        """
        Check if inner_box is inside or overlapping with outer_box
        Returns True if:
        1. inner_box is completely inside outer_box
        2. inner_box overlaps with outer_box (IoU > threshold)
        """
        # Convert boxes to [x1, y1, x2, y2] format if needed
        if hasattr(inner_box, 'cpu'):
            inner_box = inner_box.cpu().numpy()
        if hasattr(outer_box, 'cpu'):
            outer_box = outer_box.cpu().numpy()
            
        x1_i, y1_i, x2_i, y2_i = inner_box
        x1_o, y1_o, x2_o, y2_o = outer_box
        
        # Check if inner box is completely inside outer box
        if (x1_i >= x1_o and y1_i >= y1_o and 
            x2_i <= x2_o and y2_i <= y2_o):
            return True
        
        # Check for overlap using IoU
        iou = self.calculate_iou(inner_box, outer_box)
        return iou > iou_threshold
    
    def get_condition_key(self, condition_type, det1, det2):
        """Generate a unique key for a condition to track it over time"""
        # Create a key based on the condition type and approximate positions
        # This helps track the same condition across frames
        bbox1 = det1['bbox'].cpu().numpy() if hasattr(det1['bbox'], 'cpu') else det1['bbox']
        bbox2 = det2['bbox'].cpu().numpy() if hasattr(det2['bbox'], 'cpu') else det2['bbox']
        
        # Use rounded coordinates to group similar detections
        key = f"{condition_type}_{int(bbox1[0]/10)}_{int(bbox1[1]/10)}_{int(bbox2[0]/10)}_{int(bbox2[1]/10)}"
        return key
    
    def save_video_clip(self, video_path, clip_start_frame, original_filename_base, video_fps, frame_size):
        """Save a 120-second video clip centered around the detection"""
        cap = cv2.VideoCapture(video_path)
        
        # Calculate frame numbers for the clip
        rewind_frames = int(self.clip_before_duration * video_fps)
        forward_frames = int(self.clip_after_duration * video_fps)
        total_clip_frames = rewind_frames + forward_frames
        
        # Adjust start frame to ensure we have enough frames before
        actual_start_frame = max(0, clip_start_frame - rewind_frames)
        
        # Set video to start frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, actual_start_frame)
        
        # Prepare video writer
        clip_filename = os.path.join(self.save_dir, f"{original_filename_base}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(clip_filename, fourcc, video_fps, frame_size)
        
        frames_saved = 0
        
        try:
            while frames_saved < total_clip_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Save frame to clip
                out.write(frame)
                frames_saved += 1
                
                # Check if we've reached the end of the desired clip
                if frames_saved >= total_clip_frames:
                    break
        except Exception as e:
            self.logger.error(f"Error saving video clip: {e}")
        finally:
            out.release()
            cap.release()
        
        self.logger.info(f"Saved video clip: {clip_filename} ({frames_saved} frames)")
        return frames_saved
    
    def process_detections(self, result, original_frame, current_frame_number, video_fps, video_path, frame_size, cap):
        """Process detections and save images/clips only if conditions persist for validation_time"""
        if result.boxes is None:
            return None
        
        # Calculate validation frames based on validation_time and video FPS
        validation_frames = int(self.validation_time * video_fps)
        
        # Organize detections by class (filtered by confidence threshold)
        detections_by_class = {}
        for i, (xyxy, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
            confidence = float(conf)
            
            # Apply confidence threshold
            if confidence < self.confidence_threshold:
                continue
                
            class_name = result.names[int(cls)]
            
            if class_name not in detections_by_class:
                detections_by_class[class_name] = []
            
            detections_by_class[class_name].append({
                'bbox': xyxy,
                'confidence': confidence
            })
        
        # Log detected classes above threshold
        if detections_by_class:
            detected_classes = ", ".join([f"{cls}({len(dets)})" for cls, dets in detections_by_class.items()])
            self.logger.debug(f"Frame {current_frame_number}: Detections above threshold: {detected_classes}")
        
        current_conditions = set()
        
        # Check for the required overlap conditions
        # Condition 1: vacuume overlapping or in normal
        if 'vacuume' in detections_by_class and 'normal' in detections_by_class:
            for vacuume_det in detections_by_class['vacuume']:
                for normal_det in detections_by_class['normal']:
                    if self.is_inside_or_overlapping(vacuume_det['bbox'], normal_det['bbox']):
                        condition_key = self.get_condition_key("vacuume_normal", vacuume_det, normal_det)
                        current_conditions.add(condition_key)
                        
                        if condition_key not in self.condition_frames:
                            self.condition_frames[condition_key] = current_frame_number
                            self.logger.debug(f"Frame {current_frame_number}: New condition detected: {condition_key}")
                        break
                if any("vacuume_normal" in key for key in current_conditions):
                    break
        
        # Condition 2: vacuume overlapping or in suspected
        if 'vacuume' in detections_by_class and 'suspected' in detections_by_class:
            for vacuume_det in detections_by_class['vacuume']:
                for suspected_det in detections_by_class['suspected']:
                    if self.is_inside_or_overlapping(vacuume_det['bbox'], suspected_det['bbox']):
                        condition_key = self.get_condition_key("vacuume_suspected", vacuume_det, suspected_det)
                        current_conditions.add(condition_key)
                        
                        if condition_key not in self.condition_frames:
                            self.condition_frames[condition_key] = current_frame_number
                            self.logger.debug(f"Frame {current_frame_number}: New condition detected: {condition_key}")
                        break
                if any("vacuume_suspected" in key for key in current_conditions):
                    break
        
        # Condition 3: spool overlapping or in glove
        if 'spool' in detections_by_class and 'grove' in detections_by_class:
            for spool_det in detections_by_class['spool']:
                for grove_det in detections_by_class['grove']:
                    if self.is_inside_or_overlapping(spool_det['bbox'], grove_det['bbox']):
                        condition_key = self.get_condition_key("spool_grove", spool_det, grove_det)
                        current_conditions.add(condition_key)
                        
                        if condition_key not in self.condition_frames:
                            self.condition_frames[condition_key] = current_frame_number
                            self.logger.debug(f"Frame {current_frame_number}: New condition detected: {condition_key}")
                        break
                if any("spool_grove" in key for key in current_conditions):
                    break
        
        # Remove conditions that are no longer present
        expired_conditions = [k for k in self.condition_frames.keys() if k not in current_conditions]
        for condition in expired_conditions:
            self.logger.debug(f"Frame {current_frame_number}: Condition expired: {condition}")
            if condition in self.validated_conditions:
                self.validated_conditions.remove(condition)
            del self.condition_frames[condition]
        
        # Check for validated conditions (those that have persisted for validation_frames)
        should_save = False
        overlap_info = []
        validated_condition_key = None
        
        for condition_key in current_conditions:
            if condition_key in self.condition_frames:
                frames_elapsed = current_frame_number - self.condition_frames[condition_key]
                
                if frames_elapsed >= validation_frames and condition_key not in self.validated_conditions:
                    # Condition has been valid for the required number of frames
                    self.validated_conditions.add(condition_key)
                    should_save = True
                    validated_condition_key = condition_key
                    
                    # Extract condition type for logging
                    if "vacuume_normal" in condition_key:
                        overlap_info.append(f"vacuume in/overlapping normal (validated for {frames_elapsed} frames)")
                    elif "vacuume_suspected" in condition_key:
                        overlap_info.append(f"vacuume in/overlapping suspected (validated for {frames_elapsed} frames)")
                    elif "spool_grove" in condition_key:
                        overlap_info.append(f"spool in/overlapping grove (validated for {frames_elapsed} frames)")
        
        # Save images and video clip if any validated condition is met
        if should_save and validated_condition_key:
            # Generate filename base
            filename_base = self.generate_filename("O")
            
            # Save original frame
            original_filename = f"{filename_base}.jpg"
            cv2.imwrite(os.path.join(self.save_dir, original_filename), original_frame)
            
            # Save result frame with overlay
            result_frame = result.plot(line_width=self.line_thickness)
            result_filename = f"{self.generate_filename('R')}.jpg"
            cv2.imwrite(os.path.join(self.save_dir, result_filename), result_frame)
            
            self.logger.info(f"Frame {current_frame_number}: Detection saved: {original_filename}, {result_filename}")
            for info in overlap_info:
                self.logger.info(f"  - Validated condition: {info}")
            
            # Save video clip (120 seconds centered on detection)
            detection_frame = self.condition_frames[validated_condition_key]
            frames_saved = self.save_video_clip(video_path, detection_frame, filename_base, video_fps, frame_size)
            
            # Calculate where to restart detection (after the clip ends)
            rewind_frames = int(self.clip_before_duration * video_fps)
            forward_frames = int(self.clip_after_duration * video_fps)
            restart_frame = max(0, detection_frame - rewind_frames) + frames_saved
            
            self.logger.info(f"Saved video clip centered at frame {detection_frame}, restarting at frame {restart_frame}")
            
            return restart_frame
        
        return None
    
    def process_video_file(self, video_path):
        """Process a single video file completely (all frames) at maximum speed"""
        video_name = os.path.basename(video_path)
        self.logger.info(f"Processing video: {video_name}")
        
        # Reset condition tracking when starting a new video
        self.condition_frames = {}
        self.validated_conditions = set()
        
        cap = cv2.VideoCapture(video_path)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 3)
        
        if not cap.isOpened():
            self.logger.error(f"Failed to open video: {video_path}")
            return
        
        start_time = time.time()
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        video_duration = total_frames / video_fps if video_fps > 0 else 0
        
        # Get frame size for video clip saving
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (frame_width, frame_height)
        
        # Calculate validation frames based on validation_time and video FPS
        validation_frames = int(self.validation_time * video_fps)
        
        self.logger.info(f"Video details: {total_frames} frames, {video_fps:.2f} FPS, {video_duration:.2f} seconds")
        self.logger.info(f"Validation frames based on {self.validation_time}s and {video_fps:.2f} FPS: {validation_frames}")
        
        try:
            while self.running:
                ret, frame = cap.read()
                if not ret:
                    break  # End of video
                
                frame_count += 1
                current_frame_number = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                
                # YOLO inference
                results = self.model(frame, verbose=False)
                result = results[0]
                
                # Process detections
                restart_frame = self.process_detections(result, frame, current_frame_number, 
                                                       video_fps, video_path, frame_size, cap)
                
                # If a clip was saved and we need to restart at a different frame
                if restart_frame is not None and restart_frame > current_frame_number:
                    self.logger.info(f"Jumping to frame {restart_frame} after saving clip")
                    cap.set(cv2.CAP_PROP_POS_FRAMES, restart_frame)
                    
                    # Reset condition tracking for the new segment
                    self.condition_frames = {}
                    self.validated_conditions = set()
                    
                    # Continue to next frame (which will be at restart_frame)
                    continue
                
                # Calculate processing speed metrics
                elapsed = time.time() - start_time
                processing_fps = frame_count / elapsed if elapsed > 0 else 0
                
                # Display frame at maximum speed - minimal delay
                display_frame = result.plot(line_width=self.line_thickness)
                
                # Add info overlay
                cv2.putText(display_frame, f"Processing FPS: {processing_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                
                # Video information
                cv2.putText(display_frame, f"Video: {video_name}", (10, 60), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                cv2.putText(display_frame, f"Frame: {current_frame_number}/{total_frames}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                cv2.putText(display_frame, f"Video FPS: {video_fps:.2f}", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                cv2.putText(display_frame, f"Video {self.current_video_index + 1}/{len(self.video_files)}", 
                           (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                cv2.putText(display_frame, f"Confidence: {self.confidence_threshold}", 
                           (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                cv2.putText(display_frame, f"Validation: {validation_frames} frames ({self.validation_time}s)", 
                           (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 1)
                
                # Display with minimal delay for maximum speed (1ms wait)
                cv2.imshow('YOLO Video Inference', display_frame)
                
                # Use minimal wait time for maximum processing speed
                # Press 'q' to quit, space to pause/resume
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    self.running = False
                    break
                elif key == ord(' '):
                    # Pause functionality
                    while True:
                        key2 = cv2.waitKey(0) & 0xFF
                        if key2 == ord(' '):
                            break
                        elif key2 == ord('q'):
                            self.running = False
                            break
                    if not self.running:
                        break
                    
        except Exception as e:
            self.logger.error(f"Error processing video {video_name}: {e}")
        finally:
            cap.release()
            processing_time = time.time() - start_time
            self.logger.info(f"Finished processing video: {video_name}")
            self.logger.info(f"  Processed {frame_count} frames in {processing_time:.2f} seconds")
            self.logger.info(f"  Average processing FPS: {frame_count/processing_time:.2f}" if processing_time > 0 else "")
            self.logger.info(f"  Processing speed vs video FPS: {frame_count/processing_time:.2f}/{video_fps:.2f}")

    def run(self):
        """Main execution loop - process all video files completely"""
        self.logger.info(f"Starting video processing from folder: {self.video_folder}")
        
        while self.running and self.current_video_index < len(self.video_files):
            video_path = self.video_files[self.current_video_index]
            
            # Process the entire video (all frames)
            self.process_video_file(video_path)
            
            if not self.running:
                break
                
            self.current_video_index += 1
            
            if self.current_video_index < len(self.video_files):
                self.logger.info(f"Moving to next video: {self.current_video_index + 1}/{len(self.video_files)}")
        
        cv2.destroyAllWindows()
        self.logger.info("Finished processing all videos")

# Modified main execution
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    
    MODEL_PATH = r"C:\Users\RYZEN\cctv_mon\project_yolov11_obb\runs\detect\train3\weights\best.pt"
    VIDEO_FOLDER = r"C:\RecordDownload"  # Folder containing .mp4 videos
    CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence threshold
    LINE_THICKNESS = 1  # Parameter for bounding box thickness
    VALIDATION_TIME = 1.0  # Changed: Validation time in seconds (was validation_frames)
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file not found at {MODEL_PATH}")
        exit(1)
    
    if not os.path.exists(VIDEO_FOLDER):
        print(f"Error: Video folder not found at {VIDEO_FOLDER}")
        exit(1)
    
    # Create save directory if it doesn't exist
    save_dir = "vacuume-spool"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created save directory: {save_dir}")
    
    detector = DetailedVideoYOLOInference(
        MODEL_PATH, 
        VIDEO_FOLDER, 
        confidence_threshold=CONFIDENCE_THRESHOLD,
        line_thickness=LINE_THICKNESS,
        validation_time=VALIDATION_TIME  # Changed parameter to validation_time
    )
    
    try:
        detector.run()
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        cv2.destroyAllWindows()