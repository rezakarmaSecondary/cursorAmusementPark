# Amusement Park People Counting System

This system provides real-time people counting and reporting for amusement park devices using two different detection methods:

1. **Device-based Detection**: Uses microcontroller and relay to trigger people counting when a device is started
2. **Camera-based Detection**: Uses entry/exit cameras with YOLO and DeepSORT for continuous people tracking

## Features

- Real-time people counting using YOLO v11 and DeepSORT
- Multiple camera support per device
- Bounding box configuration for each camera
- Device management and status tracking
- Comprehensive reporting system
- Cooldown periods to prevent duplicate counting
- Edge case handling for crowded queues

## Tech Stack

- Backend: Python FastAPI
- Database: PostgreSQL
- Object Detection: YOLO v11
- Object Tracking: DeepSORT
- Frontend: React (separate repository)

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set up environment variables:
```bash
cp .env.example .env
# Edit .env with your configuration
```

3. Initialize database:
```bash
alembic upgrade head
```

4. Start the server:
```bash
uvicorn app.main:app --reload
```

## API Documentation

Once the server is running, visit `/docs` for the Swagger UI documentation. 