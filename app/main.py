from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError, SQLAlchemyError
from . import models, schemas
from .database import SessionLocal, engine
from .detection import DetectionSystem
from datetime import datetime, timedelta
import json
from typing import List, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create database tables
models.Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="Amusement Park Detection System",
    description="API for managing amusement park devices and camera-based detection",
    version="1.0.0"
)

# CORS middleware configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detection system
detection_system = DetectionSystem()

# Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Custom exception handlers
@app.exception_handler(IntegrityError)
async def integrity_error_handler(request, exc):
    logger.error(f"Database integrity error: {str(exc)}")
    return HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="A device with this name already exists. Please choose a different name."
    )

@app.exception_handler(SQLAlchemyError)
async def sqlalchemy_error_handler(request, exc):
    logger.error(f"Database error: {str(exc)}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An error occurred while processing your request. Please try again later."
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    logger.error(f"Unexpected error: {str(exc)}")
    return HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="An unexpected error occurred. Please try again later."
    )

@app.get("/")
async def root():
    return {"message": "Amusement Park People Counting System API"}

# Device endpoints
@app.post("/devices/", response_model=schemas.Device, status_code=status.HTTP_201_CREATED)
async def create_device(device: schemas.DeviceCreate, db: Session = Depends(get_db)):
    try:
        # Check if device with same name exists
        existing_device = db.query(models.Device).filter(models.Device.name == device.name).first()
        if existing_device:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"A device with the name '{device.name}' already exists. Please choose a different name."
            )
            
        db_device = models.Device(
            name=device.name,
            description=device.description,
            is_active=True,
            detection_method=device.detection_method,
            cooldown_seconds=device.cooldown_seconds
        )
        db.add(db_device)
        db.commit()
        db.refresh(db_device)
        logger.info(f"Created new device: {device.name}")
        return db_device
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating device: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create device. Please try again."
        )

@app.get("/devices/", response_model=List[schemas.Device])
async def list_devices(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        devices = db.query(models.Device).offset(skip).limit(limit).all()
        return devices
    except Exception as e:
        logger.error(f"Error listing devices: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve devices. Please try again."
        )

@app.get("/devices/{device_id}", response_model=schemas.Device)
async def get_device(device_id: int, db: Session = Depends(get_db)):
    try:
        device = db.query(models.Device).filter(models.Device.id == device_id).first()
        if device is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Device with ID {device_id} not found"
            )
        return device
    except Exception as e:
        logger.error(f"Error retrieving device {device_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve device. Please try again."
        )

@app.put("/devices/{device_id}", response_model=schemas.Device)
async def update_device(device_id: int, device: schemas.DeviceUpdate, db: Session = Depends(get_db)):
    try:
        db_device = db.query(models.Device).filter(models.Device.id == device_id).first()
        if db_device is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Device with ID {device_id} not found"
            )
            
        # Check if new name conflicts with existing device
        if device.name and device.name != db_device.name:
            existing_device = db.query(models.Device).filter(
                models.Device.name == device.name,
                models.Device.id != device_id
            ).first()
            if existing_device:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"A device with the name '{device.name}' already exists. Please choose a different name."
                )
        
        for key, value in device.dict(exclude_unset=True).items():
            setattr(db_device, key, value)
        db_device.updated_at = datetime.utcnow()
        db.commit()
        db.refresh(db_device)
        logger.info(f"Updated device {device_id}")
        return db_device
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating device {device_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update device. Please try again."
        )

@app.delete("/devices/{device_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_device(device_id: int, db: Session = Depends(get_db)):
    try:
        db_device = db.query(models.Device).filter(models.Device.id == device_id).first()
        if db_device is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Device with ID {device_id} not found"
            )
        db.delete(db_device)
        db.commit()
        logger.info(f"Deleted device {device_id}")
        return None
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting device {device_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete device. Please try again."
        )

# Camera endpoints
@app.post("/cameras/", response_model=schemas.Camera, status_code=status.HTTP_201_CREATED)
async def create_camera(camera: schemas.CameraCreate, db: Session = Depends(get_db)):
    try:
        # Check if device exists
        device = db.query(models.Device).filter(models.Device.id == camera.device_id).first()
        if not device:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Device with ID {camera.device_id} not found"
            )
            
        # Check if camera with same name exists for this device
        existing_camera = db.query(models.Camera).filter(
            models.Camera.device_id == camera.device_id,
            models.Camera.name == camera.name
        ).first()
        if existing_camera:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"A camera with the name '{camera.name}' already exists for this device. Please choose a different name."
            )
            
        db_camera = models.Camera(**camera.dict())
        db.add(db_camera)
        db.commit()
        db.refresh(db_camera)
        logger.info(f"Created new camera: {camera.name} for device {camera.device_id}")
        return db_camera
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error creating camera: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create camera. Please try again."
        )

@app.get("/cameras/", response_model=List[schemas.Camera])
async def list_cameras(device_id: int = None, skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    try:
        query = db.query(models.Camera)
        if device_id is not None:
            query = query.filter(models.Camera.device_id == device_id)
        cameras = query.offset(skip).limit(limit).all()
        return cameras
    except Exception as e:
        logger.error(f"Error listing cameras: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve cameras. Please try again."
        )

@app.get("/cameras/{camera_id}", response_model=schemas.Camera)
async def get_camera(camera_id: int, db: Session = Depends(get_db)):
    try:
        camera = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
        if camera is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Camera with ID {camera_id} not found"
            )
        return camera
    except Exception as e:
        logger.error(f"Error retrieving camera {camera_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve camera. Please try again."
        )

@app.put("/cameras/{camera_id}", response_model=schemas.Camera)
async def update_camera(camera_id: int, camera: schemas.CameraUpdate, db: Session = Depends(get_db)):
    try:
        db_camera = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
        if db_camera is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Camera with ID {camera_id} not found"
            )
            
        # Check if new name conflicts with existing camera for the same device
        if camera.name and camera.name != db_camera.name:
            existing_camera = db.query(models.Camera).filter(
                models.Camera.device_id == db_camera.device_id,
                models.Camera.name == camera.name,
                models.Camera.id != camera_id
            ).first()
            if existing_camera:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"A camera with the name '{camera.name}' already exists for this device. Please choose a different name."
                )
        
        for key, value in camera.dict(exclude_unset=True).items():
            setattr(db_camera, key, value)
        db.commit()
        db.refresh(db_camera)
        logger.info(f"Updated camera {camera_id}")
        return db_camera
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error updating camera {camera_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update camera. Please try again."
        )

@app.delete("/cameras/{camera_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_camera(camera_id: int, db: Session = Depends(get_db)):
    try:
        db_camera = db.query(models.Camera).filter(models.Camera.id == camera_id).first()
        if db_camera is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Camera with ID {camera_id} not found"
            )
        db.delete(db_camera)
        db.commit()
        logger.info(f"Deleted camera {camera_id}")
        return None
    except Exception as e:
        db.rollback()
        logger.error(f"Error deleting camera {camera_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to delete camera. Please try again."
        )

# Device trigger endpoint
@app.post("/device-trigger/{device_id}")
async def trigger_device(device_id: int, db: Session = Depends(get_db)):
    try:
        device = db.query(models.Device).filter(models.Device.id == device_id).first()
        if not device:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Device with ID {device_id} not found"
            )
            
        if not device.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device is not active"
            )
            
        # Check cooldown period
        last_report = db.query(models.Report).filter(
            models.Report.device_id == device_id
        ).order_by(models.Report.created_at.desc()).first()
        
        if last_report and device.cooldown_seconds > 0:
            time_since_last_report = (datetime.utcnow() - last_report.created_at).total_seconds()
            if time_since_last_report < device.cooldown_seconds:
                remaining_seconds = int(device.cooldown_seconds - time_since_last_report)
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"Device is in cooldown period. Please wait {remaining_seconds} more seconds."
                )
        
        # Process detection
        result = detection_system.process_device_trigger(device_id, db)
        
        # Create report
        report = models.Report(
            device_id=device_id,
            total_people=result["total_people"],
            detection_method="device_trigger"
        )
        db.add(report)
        db.commit()
        logger.info(f"Created report for device {device_id} with {result['total_people']} people detected")
        
        return {
            "total_people": result["total_people"],
            "processed_images": result["processed_images"]
        }
    except HTTPException:
        raise
    except Exception as e:
        db.rollback()
        logger.error(f"Error processing device trigger for device {device_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to process device trigger. Please try again."
        )

# Report endpoints
@app.get("/reports/", response_model=List[schemas.DeviceReport])
async def list_reports(
    device_id: Optional[int] = None,
    start_date: Optional[datetime] = None,
    end_date: Optional[datetime] = None,
    skip: int = 0,
    limit: int = 100,
    db: Session = Depends(get_db)
):
    try:
        query = db.query(models.DeviceReport)
        if device_id:
            query = query.filter(models.DeviceReport.device_id == device_id)
        if start_date:
            query = query.filter(models.DeviceReport.created_at >= start_date)
        if end_date:
            query = query.filter(models.DeviceReport.created_at <= end_date)
        reports = query.order_by(models.DeviceReport.created_at.desc()).offset(skip).limit(limit).all()
        return reports
    except Exception as e:
        logger.error(f"Error listing reports: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve reports. Please try again."
        )

@app.get("/reports/{report_id}", response_model=schemas.DeviceReport)
async def get_report(report_id: int, db: Session = Depends(get_db)):
    try:
        report = db.query(models.DeviceReport).filter(models.DeviceReport.id == report_id).first()
        if report is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Report with ID {report_id} not found"
            )
        return report
    except Exception as e:
        logger.error(f"Error retrieving report {report_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve report. Please try again."
        )

# Camera-based detection endpoints
@app.post("/start-camera-detection/{device_id}")
async def start_camera_detection(device_id: int, db: Session = Depends(get_db)):
    try:
        device = db.query(models.Device).filter(models.Device.id == device_id).first()
        if not device:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Device with ID {device_id} not found"
            )
            
        if not device.is_active:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Device is not active"
            )
            
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
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Both entry and exit cameras must be configured for camera-based detection"
            )
        
        # Start detection for both cameras
        detection_system.start_camera_based_detection(device_id, db)
        logger.info(f"Started camera-based detection for device {device_id}")
        return {"message": "Camera-based detection started successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error starting camera detection for device {device_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start camera-based detection. Please try again."
        )

@app.post("/stop-camera-detection/{device_id}")
async def stop_camera_detection(device_id: int):
    try:
        detection_system.stop_camera_based_detection(device_id)
        logger.info(f"Stopped camera-based detection for device {device_id}")
        return {"message": "Camera-based detection stopped successfully"}
    except Exception as e:
        logger.error(f"Error stopping camera detection for device {device_id}: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop camera-based detection. Please try again."
        ) 