# services/detector_service.py
"""FastAPI routes for emotion detection"""

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import Optional, List
import logging
import cv2
import numpy as np
from datetime import datetime

from detector import run_camera, stop_camera, list_cameras, stop_all, get_latest_frame, _active
from detector.database import DatabaseManager

logger = logging.getLogger(__name__)
router = APIRouter()

# ============================================================================
# Pydantic Models
# ============================================================================

class CameraAdd(BaseModel):
    camera_id: str
    camera_name: str
    camera_type: str  # 'webcam', 'rtsp', 'ip'
    source: str
    location_zone: str
    is_enabled: bool = True

class CameraUpdate(BaseModel):
    camera_name: Optional[str] = None
    location_zone: Optional[str] = None
    source: Optional[str] = None
    camera_type: Optional[str] = None
    is_enabled: Optional[bool] = None

class StatusResponse(BaseModel):
    status: str
    message: str
    data: Optional[dict] = None

# ============================================================================
# Camera Endpoints
# ============================================================================

@router.get("/cameras")
async def get_cameras():
    """List all cameras"""
    try:
        cameras = list_cameras()
        for cam in cameras:
            cam["status"] = "running" if cam["id"] in _active else "stopped"
        return cameras
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cameras/add", response_model=StatusResponse)
async def add_camera(camera: CameraAdd):
    """Add a new camera"""
    try:
        db = DatabaseManager()

        # Convert webcam source to int
        source = str(camera.source)

        success = db.add_camera(
            camera_id=camera.camera_id,
            camera_name=camera.camera_name,
            camera_type=camera.camera_type,
            source=source,
            location_zone=camera.location_zone,
            is_enabled=camera.is_enabled
        )

        if not success:
            raise HTTPException(status_code=500, detail="Failed to add camera")

        return StatusResponse(
            status="success",
            message=f"Camera {camera.camera_id} added successfully",
            data={"camera_id": camera.camera_id}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cameras/{camera_id}/start", response_model=StatusResponse)
async def start_camera(camera_id: str):
    """Start emotion detection for a camera"""
    try:
        result = run_camera(camera_id)
        return StatusResponse(
            status="success",
            message=f"Camera {camera_id} started",
            data=result
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cameras/{camera_id}/stop", response_model=StatusResponse)
async def stop_camera_endpoint(camera_id: str):
    """Stop a running camera"""
    try:
        result = stop_camera(camera_id)
        return StatusResponse(
            status="success",
            message=f"Camera {camera_id} stopped",
            data=result
        )
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/cameras/stop-all", response_model=StatusResponse)
async def stop_all_cameras():
    """Stop all running cameras"""
    try:
        result = stop_all()
        return StatusResponse(
            status="success",
            message="All cameras stopped",
            data=result
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/cameras/{camera_id}/status")
async def get_camera_status(camera_id: str):
    """Get status of a specific camera"""
    try:
        cameras = list_cameras()
        camera = next((c for c in cameras if c["id"] == camera_id), None)

        if not camera:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

        return {
            "camera_id": camera_id,
            "name": camera["name"],
            "zone": camera["location_zone"],
            "enabled": camera["enabled"],
            "status": "running" if camera_id in _active else "stopped",
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.patch("/cameras/{camera_id}", response_model=StatusResponse)
async def update_camera(camera_id: str, updates: CameraUpdate):
    """Update camera info"""
    try:
        db = DatabaseManager()

        update_dict = updates.dict(exclude_none=True)
        if not update_dict:
            raise HTTPException(status_code=400, detail="No fields to update")

        success = db.update_camera(camera_id, update_dict)
        if not success:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

        return StatusResponse(
            status="success",
            message=f"Camera {camera_id} updated"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/cameras/{camera_id}", response_model=StatusResponse)
async def delete_camera(camera_id: str):
    """Delete a camera"""
    try:
        # Stop if running
        if camera_id in _active:
            stop_camera(camera_id)

        db = DatabaseManager()
        success = db.delete_camera(camera_id)

        if not success:
            raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

        return StatusResponse(
            status="success",
            message=f"Camera {camera_id} deleted"
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Streaming Endpoint
# ============================================================================

def generate_mjpeg_stream(camera_id: str):
    """Generate MJPEG stream from frame cache"""
    import time

    try:
        while True:
            frame = get_latest_frame(camera_id)

            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "Waiting for frames...", (150, 240),
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

            ret, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                time.sleep(0.033)
                continue

            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' +
                   jpeg.tobytes() +
                   b'\r\n')

            time.sleep(0.033)

    except GeneratorExit:
        logger.info(f"Stream closed for {camera_id}")
    except Exception as e:
        logger.error(f"Stream error for {camera_id}: {e}")


@router.get("/stream/{camera_id}")
async def stream_camera(camera_id: str):
    """Stream live video from a running camera"""
    cameras = list_cameras()
    camera = next((c for c in cameras if c["id"] == camera_id), None)

    if not camera:
        raise HTTPException(status_code=404, detail=f"Camera {camera_id} not found")

    if camera_id not in _active:
        raise HTTPException(status_code=400, detail=f"Camera {camera_id} is not running")

    return StreamingResponse(
        generate_mjpeg_stream(camera_id),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


# ============================================================================
# Screenshot Endpoints
# ============================================================================

@router.post("/cameras/{camera_id}/screenshot", response_model=StatusResponse)
async def take_manual_screenshot(camera_id: str):
    """Take a manual screenshot from a running camera"""
    try:
        if camera_id not in _active:
            raise HTTPException(status_code=400, detail=f"Camera {camera_id} is not running")

        frame = get_latest_frame(camera_id)
        if frame is None:
            raise HTTPException(status_code=404, detail="No frame available")

        timestamp = datetime.now()
        timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"manual_{camera_id}_{timestamp_str}.jpg"

        # Encode frame
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not ret:
            raise HTTPException(status_code=500, detail="Failed to encode frame")

        file_bytes = buffer.tobytes()

        # Upload to Supabase Storage
        db = DatabaseManager()
        public_url, storage_path = db.upload_screenshot(file_bytes, filename)

        if not public_url:
            raise HTTPException(status_code=500, detail="Failed to upload screenshot")

        # Save to database
        db.insert_screenshot(
            camera_id=camera_id,
            face_id="manual",
            emotion="N/A",
            confidence=0.0,
            filename=filename,
            storage_path=storage_path,
            public_url=public_url,
            file_size_bytes=len(file_bytes),
            trigger_reason="manual_capture"
        )

        return StatusResponse(
            status="success",
            message="Screenshot captured",
            data={
                "filename": filename,
                "public_url": public_url,
                "timestamp": timestamp_str
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screenshots")
async def list_screenshots(
    camera_id: Optional[str] = None,
    limit: int = 50,
    offset: int = 0
):
    """List all screenshots"""
    try:
        db = DatabaseManager()
        screenshots = db.get_screenshots(camera_id=camera_id, limit=limit, offset=offset)
        total = db.get_screenshot_count(camera_id=camera_id)
        return {
            "screenshots": screenshots,
            "total": total,
            "limit": limit,
            "offset": offset
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/screenshots/{screenshot_id}")
async def get_screenshot(screenshot_id: str):
    """Get a single screenshot"""
    try:
        db = DatabaseManager()
        screenshot = db.get_screenshot_by_id(screenshot_id)
        if not screenshot:
            raise HTTPException(status_code=404, detail="Screenshot not found")
        return screenshot
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/screenshots/{screenshot_id}", response_model=StatusResponse)
async def delete_screenshot(screenshot_id: str):
    """Delete a screenshot from database and storage"""
    try:
        db = DatabaseManager()
        screenshot = db.delete_screenshot(screenshot_id)

        if not screenshot:
            raise HTTPException(status_code=404, detail="Screenshot not found")

        # Delete from Supabase Storage
        if screenshot.get("storage_path"):
            db.delete_screenshot_file(screenshot["storage_path"])

        return StatusResponse(
            status="success",
            message=f"Screenshot {screenshot_id} deleted",
            data={"filename": screenshot.get("filename")}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ============================================================================
# Health Check
# ============================================================================

@router.get("/health")
async def health_check():
    """Health check for detector service"""
    return {
        "status": "healthy",
        "service": "detector",
        "active_cameras": len(_active),
        "active_camera_ids": list(_active.keys()),
        "timestamp": datetime.now().isoformat()
    }