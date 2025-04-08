import cv2
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from sqlalchemy.orm import Session
from datetime import datetime, timedelta
import json
from . import models
import threading
import time
import os
from pathlib import Path
import torch
from ultralytics.nn.tasks import DetectionModel
from dotenv import load_dotenv
import logging
from typing import List, Tuple, Optional
import asyncio

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

class DetectionSystem:
    def __init__(self):
        # Add DetectionModel to safe globals
        torch.serialization.add_safe_globals([DetectionModel])
        
        self.model_path = os.getenv('YOLO_MODEL_PATH', 'models/best.pt')
        self.confidence_threshold = float(os.getenv('CONFIDENCE_THRESHOLD', 0.5))
        self.iou_threshold = float(os.getenv('IOU_THRESHOLD', 0.5))
        self.detection_interval = int(os.getenv('DETECTION_INTERVAL', 5))
        
        # Initialize YOLO model
        try:
            logger.info(f"Loading YOLO model from {self.model_path}")
            self.model = YOLO(self.model_path)
            logger.info("YOLO model loaded successfully")
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            raise
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=30,
            n_init=50,
            nms_max_overlap=1.0,
            max_cosine_distance=0.3,
            nn_budget=None,
            override_track_class=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True
        )
        
        # Store active detection tasks
        self.active_detections = {}
        
        self.tracked_people = {}  # Store tracked people with their last seen time
        self.min_reappearance_time = 600  # 10 minutes in seconds
        self.active_cameras = {}
        self.camera_threads = {}
        self.detection_results = {}
        self.image_save_path = Path("detection_images")
        self.image_save_path.mkdir(exist_ok=True)

    def process_device_trigger(self, device_id: int, db: Session) -> dict:
        try:
            # Get device cameras
            device = db.query(models.Device).filter(models.Device.id == device_id).first()
            if not device:
                raise ValueError(f"Device with ID {device_id} not found")

            cameras = db.query(models.Camera).filter(models.Camera.device_id == device_id).all()
            if not cameras:
                raise ValueError(f"No cameras found for device {device_id}")

            total_people = 0
            processed_images = []

            for camera in cameras:
                # Initialize video capture
                cap = cv2.VideoCapture(camera.rtsp_url)
                if not cap.isOpened():
                    logger.error(f"Failed to open RTSP stream for camera {camera.id}")
                    continue

                try:
                    # Set buffer size to 1 to get the latest frame
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Skip frames to get to the latest one
                    for _ in range(10):  # Skip more frames to ensure we get a fresh one
                        cap.grab()
                    
                    # Read the latest frame
                    ret, frame = cap.read()
                    if not ret:
                        logger.error(f"Failed to read frame from camera {camera.id}")
                        continue

                    # Process the frame with YOLO and save the result
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_filename = f"device_{device_id}_camera_{camera.id}_{timestamp}.jpg"
                    image_path = self.image_save_path / image_filename
                    
                    # Run YOLO detection and save the result
                    results = self.model(frame, save=True, project=str(self.image_save_path), name=image_filename)
                    
                    # Get the number of people detected
                    people_count = len(results[0].boxes)
                    total_people += people_count
                    
                    # Add the saved image path to the list
                    processed_images.append(str(image_path))
                    
                    # Log frame capture details
                    logger.info(f"Captured frame for camera {camera.id} at {timestamp} with {people_count} people detected")

                finally:
                    # Always release the capture
                    cap.release()
                    # Force garbage collection
                    import gc
                    gc.collect()
                    # Add a small delay before processing next camera
                    time.sleep(0.5)

            return {
                "total_people": total_people,
                "processed_images": processed_images
            }

        except Exception as e:
            logger.error(f"Error processing device trigger: {str(e)}")
            raise

    def start_camera_based_detection(self, device_id: int, db: Session):
        """Start continuous camera-based detection for a device"""
        device = db.query(models.Device).filter(models.Device.id == device_id).first()
        if not device or device.detection_method != "camera_based":
            return False

        cameras = db.query(models.Camera).filter(
            models.Camera.device_id == device_id,
            models.Camera.is_active == True
        ).all()

        for camera in cameras:
            if camera.id not in self.active_cameras:
                thread = threading.Thread(
                    target=self._camera_detection_loop,
                    args=(camera, db)
                )
                thread.daemon = True
                self.camera_threads[camera.id] = thread
                self.active_cameras[camera.id] = True
                thread.start()

        return True

    def stop_camera_based_detection(self, device_id: int):
        """Stop camera-based detection for a device"""
        for camera_id in list(self.active_cameras.keys()):
            self.active_cameras[camera_id] = False
            if camera_id in self.camera_threads:
                self.camera_threads[camera_id].join()
                del self.camera_threads[camera_id]

    def _camera_detection_loop(self, camera: models.Camera, db: Session):
        """Continuous detection loop for a camera"""
        cap = cv2.VideoCapture(camera.rtsp_url)
        
        while self.active_cameras.get(camera.id, False):
            ret, frame = cap.read()
            if not ret:
                time.sleep(1)
                continue

            # Process frame
            self._process_frame(frame, camera)

            # Sleep briefly to prevent high CPU usage
            time.sleep(0.1)

        cap.release()

    def _process_frame(self, frame, camera: models.Camera) -> int:
        """Process a single frame for people detection"""
        # Get bounding box from camera configuration
        x1 = int(camera.bounding_box_x1 * frame.shape[1])
        y1 = int(camera.bounding_box_y1 * frame.shape[0])
        x2 = int(camera.bounding_box_x2 * frame.shape[1])
        y2 = int(camera.bounding_box_y2 * frame.shape[0])
        
        # Crop frame to bounding box
        cropped_frame = frame[y1:y2, x1:x2]

        # Run YOLO detection
        results = self.model(cropped_frame, conf=self.confidence_threshold, iou=self.iou_threshold, verbose=False)
        detections = []

        for result in results:
            boxes = result.boxes.xyxy.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            
            for box, conf, class_id in zip(boxes, confidences, class_ids):
                detections.append((box, conf, class_id))

        # Update tracker
        tracks = self.tracker.update_tracks(detections, frame=cropped_frame)

        # Process tracks
        current_time = datetime.now()
        people_count = 0

        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()

            # Check if person was recently counted
            if track_id in self.tracked_people:
                last_seen = self.tracked_people[track_id]
                if (current_time - last_seen).total_seconds() < self.min_reappearance_time:
                    continue

            # Update tracking information
            self.tracked_people[track_id] = current_time
            people_count += 1

        return people_count

    def cleanup_old_tracks(self):
        """Clean up old tracking data"""
        current_time = datetime.now()
        old_tracks = [
            track_id for track_id, last_seen in self.tracked_people.items()
            if (current_time - last_seen).total_seconds() > self.min_reappearance_time
        ]
        for track_id in old_tracks:
            del self.tracked_people[track_id]

    def process_frame(self, frame: np.ndarray) -> Tuple[List, List]:
        """Process a single frame and return detections and tracks."""
        try:
            # Run YOLO detection
            results = self.model(
                frame,
                conf=self.confidence_threshold,
                iou=self.iou_threshold,
                verbose=False
            )
            
            # Extract detections
            detections = []
            for result in results:
                boxes = result.boxes.xyxy.cpu().numpy()
                confidences = result.boxes.conf.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                
                for box, conf, class_id in zip(boxes, confidences, class_ids):
                    detections.append((box, conf, class_id))
            
            # Update tracker with detections
            tracks = self.tracker.update_tracks(detections, frame=frame)
            
            return detections, tracks
            
        except Exception as e:
            logger.error(f"Error processing frame: {str(e)}")
            return [], []
            
    async def process_camera(self, device_id: int, rtsp_url: str, camera_type: str):
        """Process frames from a camera stream."""
        try:
            cap = cv2.VideoCapture(rtsp_url)
            if not cap.isOpened():
                logger.error(f"Failed to open camera stream: {rtsp_url}")
                return
                
            logger.info(f"Started processing camera stream: {rtsp_url}")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.error("Failed to read frame from camera")
                    break
                    
                # Process frame
                detections, tracks = self.process_frame(frame)
                
                # Log detection results
                logger.info(f"Device {device_id} - {camera_type} camera: {len(tracks)} people detected")
                
                # Add delay between detections
                await asyncio.sleep(self.detection_interval)
                
        except Exception as e:
            logger.error(f"Error in camera processing: {str(e)}")
        finally:
            if 'cap' in locals():
                cap.release()
                
    def start_detection(self, device_id: int, rtsp_url: str, camera_type: str):
        """Start camera-based detection for a device."""
        if device_id in self.active_detections:
            logger.warning(f"Detection already running for device {device_id}")
            return
            
        # Create and store detection task
        task = asyncio.create_task(
            self.process_camera(device_id, rtsp_url, camera_type)
        )
        self.active_detections[device_id] = task
        
    def stop_detection(self, device_id: int):
        """Stop camera-based detection for a device."""
        if device_id in self.active_detections:
            self.active_detections[device_id].cancel()
            del self.active_detections[device_id]
            logger.info(f"Stopped detection for device {device_id}")
        else:
            logger.warning(f"No active detection found for device {device_id}")
            
    def get_active_detections(self) -> List[int]:
        """Get list of device IDs with active detections."""
        return list(self.active_detections.keys()) 