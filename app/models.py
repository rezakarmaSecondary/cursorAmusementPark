from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float, JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database import Base
from datetime import datetime
import json

class Device(Base):
    __tablename__ = "devices"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String, unique=True, index=True)
    description = Column(String, nullable=True)
    detection_method = Column(String)  # 'device_trigger' or 'camera_based'
    cooldown_seconds = Column(Integer, default=0)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    cameras = relationship("Camera", back_populates="device", cascade="all, delete-orphan")
    reports = relationship("Report", back_populates="device", cascade="all, delete-orphan")
    person_detections = relationship("PersonDetection", back_populates="device")

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)
    name = Column(String, nullable=False)
    rtsp_url = Column(String, nullable=False)
    camera_type = Column(String, nullable=False)  # 'entry', 'exit', or 'monitoring'
    polygon_points = Column(JSON, nullable=True)  # Store polygon points as JSON array
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, onupdate=datetime.utcnow)

    device = relationship("Device", back_populates="cameras")
    person_detections = relationship("PersonDetection", back_populates="camera")

    @property
    def bounding_box(self):
        return {
            "x1": self.bounding_box_x1,
            "y1": self.bounding_box_y1,
            "x2": self.bounding_box_x2,
            "y2": self.bounding_box_y2
        }

    @bounding_box.setter
    def bounding_box(self, value):
        if isinstance(value, dict):
            self.bounding_box_x1 = value["x1"]
            self.bounding_box_y1 = value["y1"]
            self.bounding_box_x2 = value["x2"]
            self.bounding_box_y2 = value["y2"]
        elif isinstance(value, (list, tuple)) and len(value) == 4:
            self.bounding_box_x1 = value[0]
            self.bounding_box_y1 = value[1]
            self.bounding_box_x2 = value[2]
            self.bounding_box_y2 = value[3]

class PersonDetection(Base):
    __tablename__ = "person_detections"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)
    camera_id = Column(Integer, ForeignKey("cameras.id"), nullable=False)
    person_id = Column(String, nullable=False)  # Unique ID for each person from DeepSORT
    detection_time = Column(DateTime, default=datetime.utcnow)
    confidence = Column(Float, nullable=False)
    x1 = Column(Float, nullable=True)  # Bounding box coordinates
    y1 = Column(Float, nullable=True)
    x2 = Column(Float, nullable=True)
    y2 = Column(Float, nullable=True)
    is_entry = Column(Boolean, default=False)
    is_exit = Column(Boolean, default=False)
    status = Column(String, default='detected')  # detected, tracked, counted, lost
    last_seen = Column(DateTime, nullable=True)
    is_counted = Column(Boolean, default=False)  # Whether this detection has been counted in reports

    # Relationships
    device = relationship("Device", back_populates="person_detections")
    camera = relationship("Camera", back_populates="person_detections")

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey("devices.id"), nullable=False)
    start_time = Column(DateTime, nullable=False)
    end_time = Column(DateTime, nullable=False)
    total_entered = Column(Integer, default=0)
    total_exited = Column(Integer, default=0)
    current_count = Column(Integer, default=0)  # Current number of people in the area
    detection_method = Column(String, nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    # Relationships
    device = relationship("Device", back_populates="reports") 