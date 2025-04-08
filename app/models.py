from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Float
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

class Camera(Base):
    __tablename__ = "cameras"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey("devices.id"))
    name = Column(String)
    rtsp_url = Column(String)
    camera_type = Column(String)  # "entry" or "exit"
    bounding_box_x1 = Column(Float)  # Left coordinate (0-1)
    bounding_box_y1 = Column(Float)  # Top coordinate (0-1)
    bounding_box_x2 = Column(Float)  # Right coordinate (0-1)
    bounding_box_y2 = Column(Float)  # Bottom coordinate (0-1)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, nullable=True)

    device = relationship("Device", back_populates="cameras")
    detections = relationship("PersonDetection", back_populates="camera")

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
    camera_id = Column(Integer, ForeignKey("cameras.id"))
    person_id = Column(String)  # DeepSORT tracking ID
    confidence = Column(Float)
    bbox = Column(String)  # JSON string of [x1, y1, x2, y2]
    timestamp = Column(DateTime(timezone=True), server_default=func.now())
    is_counted = Column(Boolean, default=False)
    last_seen = Column(DateTime(timezone=True))

    camera = relationship("Camera", back_populates="detections")

class Report(Base):
    __tablename__ = "reports"

    id = Column(Integer, primary_key=True, index=True)
    device_id = Column(Integer, ForeignKey("devices.id"))
    total_people = Column(Integer)
    detection_method = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    device = relationship("Device", back_populates="reports") 