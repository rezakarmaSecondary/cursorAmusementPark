from pydantic import BaseModel, Field, confloat
from typing import Optional, List, Tuple
from datetime import datetime
from enum import Enum

class DetectionMethod(str, Enum):
    device_trigger = "device_trigger"
    camera_based = "camera_based"
    both = "both"

class BoundingBox(BaseModel):
    x1: confloat(ge=0.0, le=1.0)  # Left coordinate (0-1)
    y1: confloat(ge=0.0, le=1.0)  # Top coordinate (0-1)
    x2: confloat(ge=0.0, le=1.0)  # Right coordinate (0-1)
    y2: confloat(ge=0.0, le=1.0)  # Bottom coordinate (0-1)

    def to_tuple(self) -> Tuple[float, float, float, float]:
        return (self.x1, self.y1, self.x2, self.y2)

    @classmethod
    def from_tuple(cls, coords: Tuple[float, float, float, float]):
        return cls(x1=coords[0], y1=coords[1], x2=coords[2], y2=coords[3])

class DeviceBase(BaseModel):
    name: str
    description: Optional[str] = None
    detection_method: DetectionMethod
    cooldown_seconds: int = Field(ge=0, description="Cooldown period in seconds between detections")
    is_active: bool = True

    class Config:
        from_attributes = True

class DeviceCreate(DeviceBase):
    pass

class DeviceUpdate(BaseModel):
    name: Optional[str] = None
    description: Optional[str] = None
    detection_method: Optional[DetectionMethod] = None
    cooldown_seconds: Optional[int] = Field(None, ge=0, description="Cooldown period in seconds between detections")
    is_active: Optional[bool] = None

    class Config:
        from_attributes = True

class Device(DeviceBase):
    id: int
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class CameraBase(BaseModel):
    device_id: int
    name: str
    rtsp_url: str
    camera_type: str
    bounding_box: BoundingBox

class CameraCreate(CameraBase):
    pass

class CameraUpdate(BaseModel):
    name: Optional[str] = None
    rtsp_url: Optional[str] = None
    camera_type: Optional[str] = None
    bounding_box: Optional[BoundingBox] = None
    is_active: Optional[bool] = None

class Camera(CameraBase):
    id: int
    is_active: bool
    created_at: datetime
    updated_at: Optional[datetime] = None

    class Config:
        from_attributes = True

class PersonDetectionBase(BaseModel):
    camera_id: int
    person_id: str
    confidence: float
    bbox: str
    is_counted: bool = False
    last_seen: Optional[datetime] = None

class PersonDetectionCreate(PersonDetectionBase):
    pass

class PersonDetection(PersonDetectionBase):
    id: int
    timestamp: datetime

    class Config:
        orm_mode = True

class DeviceReportBase(BaseModel):
    device_id: int
    start_time: datetime
    end_time: datetime
    total_people: int
    detection_method: str

class DeviceReportCreate(DeviceReportBase):
    pass

class DeviceReport(DeviceReportBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

class ReportBase(BaseModel):
    device_id: int
    total_people: int
    detection_method: str

class ReportCreate(ReportBase):
    pass

class Report(ReportBase):
    id: int
    created_at: datetime

    class Config:
        from_attributes = True 