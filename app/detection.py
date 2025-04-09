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
        """Initialize the detection system."""
        # Load YOLO model
        self.model = YOLO('models/best.pt')  # Using custom trained model
        
        # Initialize DeepSORT tracker
        self.tracker = DeepSort(
            max_age=30,
            n_init=3,
            max_iou_distance=0.7,
            max_cosine_distance=0.3,
            nn_budget=None,
            embedder="mobilenet",
            half=True,
            bgr=True,
            embedder_gpu=True
        )
        
        # Initialize tracking data
        self.tracking_data = {}  # device_id -> tracking info
        self.detection_threads = {}  # device_id -> threads
        self.tracking_lock = threading.Lock()
        
        # Create directory for saving images
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
        """Start continuous camera-based detection for a device."""
        try:
            # Get device and its cameras
            device = db.query(models.Device).filter(models.Device.id == device_id).first()
            if not device:
                raise ValueError(f"Device with ID {device_id} not found")

            # Get entry and exit cameras
            entry_camera = db.query(models.Camera).filter(
                models.Camera.device_id == device_id,
                models.Camera.camera_type == "entry"
            ).first()
            
            exit_camera = db.query(models.Camera).filter(
                models.Camera.device_id == device_id,
                models.Camera.camera_type == "exit"
            ).first()

            if not entry_camera or not exit_camera:
                raise ValueError("Both entry and exit cameras must be configured for camera-based detection")

            # Initialize tracking data for this device
            self.tracking_data[device_id] = {
                "entry_tracks": {},  # track_id -> last_seen_time
                "exit_tracks": {},   # track_id -> last_seen_time
                "current_count": 0,
                "last_report_time": datetime.now(),
                "is_running": True,
                "cooldown_period": 600  # 10 minutes in seconds
            }

            # Start detection threads for both cameras
            self.detection_threads[device_id] = {
                "entry": threading.Thread(
                    target=self._process_camera_stream,
                    args=(entry_camera, "entry", device_id, db)
                ),
                "exit": threading.Thread(
                    target=self._process_camera_stream,
                    args=(exit_camera, "exit", device_id, db)
                )
            }

            # Start both threads
            self.detection_threads[device_id]["entry"].start()
            self.detection_threads[device_id]["exit"].start()

            logger.info(f"Started camera-based detection for device {device_id}")

        except Exception as e:
            logger.error(f"Error starting camera detection: {str(e)}")
            raise

    def _process_camera_stream(self, camera: models.Camera, camera_type: str, device_id: int, db: Session):
        """Process camera stream and update tracking data."""
        max_retries = 5
        retry_count = 0
        retry_delay = 3  # seconds
        
        while self.tracking_data[device_id]["is_running"] and retry_count < max_retries:
            try:
                cap = cv2.VideoCapture(camera.rtsp_url)
                if not cap.isOpened():
                    logger.error(f"Failed to open RTSP stream for camera {camera.id}")
                    retry_count += 1
                    time.sleep(retry_delay)
                    continue

                logger.info(f"Successfully connected to camera {camera.id}")
                retry_count = 0  # Reset retry counter on success

                try:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    last_process_time = time.time()

                    while self.tracking_data[device_id]["is_running"]:
                        # Maintain processing rate
                        if time.time() - last_process_time < 0.1:  # 10 FPS
                            continue
                        last_process_time = time.time()

                        # Skip frames
                        for _ in range(5):
                            cap.grab()

                        # Read frame
                        ret, frame = cap.read()
                        if not ret or frame is None or frame.size == 0:
                            logger.warning(f"Invalid frame from camera {camera.id}")
                            continue

                        # YOLO processing
                        results = self.model(frame)
                        
                        # Detection processing
                        detections = []
                        for result in results:
                            boxes = result.boxes
                            for box in boxes:
                                if box.cls[0] == 0:  # Person class
                                    try:
                                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                        conf = box.conf[0].item()
                                        width = x2 - x1
                                        height = y2 - y1
                                        
                                        # Validate coordinates
                                        if all(coord >= 0 for coord in [x1, y1, width, height]):
                                            detections.append(([x1, y1, width, height], conf))
                                    except Exception as e:
                                        logger.error(f"Error processing box: {str(e)}")

                        # Update tracker
                        if detections:
                            detections_list = [
                                (
                                    [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])],
                                    float(conf)
                                ) for (bbox, conf) in detections
                            ]
                            tracks = self.tracker.update_tracks(detections_list, frame=frame)
                        else:
                            tracks = []

                        # Process tracks and update counting
                        current_time = datetime.now()
                        with self.tracking_lock:
                            # Get the tracks dictionary for this camera type
                            tracks_dict = self.tracking_data[device_id][f"{camera_type}_tracks"]
                            
                            # Update existing tracks and check for new entries/exits
                            active_tracks = set()
                            for track in tracks:
                                if not track.is_confirmed():
                                    continue
                                    
                                track_id = track.track_id
                                active_tracks.add(track_id)
                                
                                if track_id not in tracks_dict:
                                    # New track detected
                                    tracks_dict[track_id] = current_time
                                    if camera_type == "entry":
                                        self.tracking_data[device_id]["current_count"] += 1
                                else:
                                    # Existing track, update last seen time
                                    tracks_dict[track_id] = current_time
                            
                            # Check for tracks that are no longer active
                            for track_id in list(tracks_dict.keys()):
                                if track_id not in active_tracks:
                                    # Track is no longer visible
                                    last_seen = tracks_dict[track_id]
                                    time_since_last_seen = (current_time - last_seen).total_seconds()
                                    
                                    if time_since_last_seen > self.tracking_data[device_id]["cooldown_period"]:
                                        # Remove track after cooldown period
                                        del tracks_dict[track_id]
                                        if camera_type == "exit":
                                            self.tracking_data[device_id]["current_count"] -= 1

                            # Create report if enough time has passed
                            if (current_time - self.tracking_data[device_id]["last_report_time"]).total_seconds() >= 60:
                                self._create_camera_report(device_id, db)
                                self.tracking_data[device_id]["last_report_time"] = current_time

                except Exception as e:
                    logger.error(f"Processing error for camera {camera.id}: {str(e)}")
                finally:
                    cap.release()
                    time.sleep(1)

            except Exception as e:
                logger.error(f"Connection error for camera {camera.id}: {str(e)}")
                retry_count += 1
                time.sleep(retry_delay)

        if retry_count >= max_retries:
            logger.error(f"Max retries ({max_retries}) reached for camera {camera.id}")

    def _create_camera_report(self, device_id: int, db: Session):
        """Create a report for camera-based detection."""
        try:
            tracking_data = self.tracking_data[device_id]
            
            # Create new report
            report = models.Report(
                device_id=device_id,
                total_people=tracking_data["current_count"],
                detection_method="camera_based"
            )
            
            db.add(report)
            db.commit()
            
            logger.info(f"Created camera-based report for device {device_id}: "
                       f"Total={tracking_data['current_count']}")

        except Exception as e:
            logger.error(f"Error creating camera report: {str(e)}")
            db.rollback()

    def stop_camera_based_detection(self, device_id: int):
        """Stop camera-based detection for a device."""
        try:
            if device_id in self.tracking_data:
                # Stop the detection threads
                self.tracking_data[device_id]["is_running"] = False
                
                # Wait for threads to finish
                if device_id in self.detection_threads:
                    self.detection_threads[device_id]["entry"].join()
                    self.detection_threads[device_id]["exit"].join()
                    
                    # Clean up
                    del self.detection_threads[device_id]
                    del self.tracking_data[device_id]
                
                logger.info(f"Stopped camera-based detection for device {device_id}")
                
        except Exception as e:
            logger.error(f"Error stopping camera detection: {str(e)}")
            raise

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