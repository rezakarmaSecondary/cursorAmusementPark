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
                # Initialize video capture with FFMPEG backend
                rtsp_url = camera.rtsp_url
                if not rtsp_url.startswith('rtsp'):
                    rtsp_url += '?tcp'  # Add TCP transport if needed
                
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    logger.error(f"Failed to open RTSP stream for camera {camera.id}")
                    continue

                try:
                    # Set buffer size to 1 to get the latest frame
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                    
                    # Skip frames to get to the latest one
                    for _ in range(10):
                        cap.grab()
                    
                    # Read the latest frame
                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        logger.error(f"Failed to read valid frame from camera {camera.id}")
                        continue

                    # Process the frame with YOLO and save the result
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_filename = f"device_{device_id}_camera_{camera.id}_{timestamp}.jpg"
                    image_path = self.image_save_path / image_filename
                    
                    # Run YOLO detection and save the result
                    results = self.model(frame, save=True, project=str(self.image_save_path), name=image_filename)
                    
                    # Get the number of people detected
                    if results and len(results[0].boxes) > 0:
                        people_count = len(results[0].boxes)
                    else:
                        people_count = 0
                        
                    total_people += people_count
                    
                    # Add the saved image path to the list
                    processed_images.append(str(image_path))
                    
                    logger.info(f"Captured frame for camera {camera.id} at {timestamp} with {people_count} people detected")

                finally:
                    # Release resources
                    cap.release()
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
            device = db.query(models.Device).filter(models.Device.id == device_id).first()
            if not device:
                raise ValueError(f"Device with ID {device_id} not found")

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

            self.tracking_data[device_id] = {
                "entry_tracks": {},
                "exit_tracks": {},
                "current_count": 0,
                "last_report_time": datetime.now(),
                "is_running": True,
                "cooldown_period": 600
            }

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

            self.detection_threads[device_id]["entry"].start()
            self.detection_threads[device_id]["exit"].start()

            logger.info(f"Started camera-based detection for device {device_id}")

        except Exception as e:
            logger.error(f"Error starting camera detection: {str(e)}")
            raise

    def _process_camera_stream(self, camera: models.Camera, camera_type: str, device_id: int, db: Session):
        """Process camera stream with reconnection logic."""
        max_retries = 5
        reconnect_attempts = 0
        max_reconnect_attempts = 3
        error_count = 0
        max_errors_before_reconnect = 10
        
        rtsp_url = camera.rtsp_url
        if not rtsp_url.startswith('rtsp'):
            rtsp_url += '?tcp'

        while self.tracking_data.get(device_id, {}).get("is_running", False) and reconnect_attempts < max_reconnect_attempts:
            cap = None
            try:
                cap = cv2.VideoCapture(rtsp_url, cv2.CAP_FFMPEG)
                if not cap.isOpened():
                    logger.error(f"Failed to open RTSP stream for camera {camera.id}")
                    reconnect_attempts += 1
                    time.sleep(2)
                    continue

                logger.info(f"Successfully connected to camera {camera.id}")
                reconnect_attempts = 0
                error_count = 0

                while self.tracking_data[device_id]["is_running"]:
                    # Skip frames to get the latest one
                    for _ in range(5):
                        cap.grab()

                    ret, frame = cap.read()
                    if not ret or frame is None or frame.size == 0:
                        error_count += 1
                        if error_count >= max_errors_before_reconnect:
                            logger.error("Too many consecutive errors, reconnecting...")
                            break
                        continue
                    
                    error_count = 0  # Reset error counter

                    # Process frame
                    results = self.model(frame)
                    
                    # Filter detections
                    detections = []
                    for result in results:
                        boxes = result.boxes
                        for box in boxes:
                            if box.cls[0] == 0:
                                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                                conf = box.conf[0].item()
                                width = x2 - x1
                                height = y2 - y1
                                detections.append(([x1, y1, width, height], conf))

                    # Update tracker
                    if detections:
                        detections_list = [
                            (
                                [float(det[0][0]), float(det[0][1]),
                                 float(det[0][2]), float(det[0][3])],
                                float(det[1])
                            ) for det in detections
                        ]
                        tracks = self.tracker.update_tracks(detections_list, frame=frame)
                    else:
                        tracks = []

                    # Update tracking data
                    current_time = datetime.now()
                    with self.tracking_lock:
                        tracks_dict = self.tracking_data[device_id][f"{camera_type}_tracks"]
                        active_tracks = set()
                        
                        for track in tracks:
                            if not track.is_confirmed():
                                continue
                                
                            track_id = track.track_id
                            active_tracks.add(track_id)
                            
                            if track_id not in tracks_dict:
                                tracks_dict[track_id] = current_time
                                if camera_type == "entry":
                                    self.tracking_data[device_id]["current_count"] += 1
                            else:
                                tracks_dict[track_id] = current_time
                        
                        # Cleanup old tracks
                        for track_id in list(tracks_dict.keys()):
                            if track_id not in active_tracks:
                                last_seen = tracks_dict[track_id]
                                if (current_time - last_seen).total_seconds() > self.tracking_data[device_id]["cooldown_period"]:
                                    del tracks_dict[track_id]
                                    if camera_type == "exit":
                                        self.tracking_data[device_id]["current_count"] -= 1

                        # Create report
                        if (current_time - self.tracking_data[device_id]["last_report_time"]).total_seconds() >= 60:
                            self._create_camera_report(device_id, db)
                            self.tracking_data[device_id]["last_report_time"] = current_time

                    time.sleep(0.1)

            except Exception as e:
                logger.error(f"Error processing camera stream: {str(e)}")
            finally:
                if cap is not None:
                    cap.release()
                time.sleep(2)
                reconnect_attempts += 1

        logger.warning(f"Stopped processing for camera {camera.id}")

    def _create_camera_report(self, device_id: int, db: Session):
        """Create a report for camera-based detection."""
        try:
            tracking_data = self.tracking_data[device_id]
            report = models.Report(
                device_id=device_id,
                total_people=tracking_data["current_count"],
                detection_method="camera_based"
            )
            db.add(report)
            db.commit()
            logger.info(f"Created camera-based report for device {device_id}: Total={tracking_data['current_count']}")
        except Exception as e:
            logger.error(f"Error creating camera report: {str(e)}")
            db.rollback()

    def stop_camera_based_detection(self, device_id: int):
        """Stop camera-based detection for a device."""
        try:
            if device_id in self.tracking_data:
                self.tracking_data[device_id]["is_running"] = False
                
                if device_id in self.detection_threads:
                    self.detection_threads[device_id]["entry"].join()
                    self.detection_threads[device_id]["exit"].join()
                    
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