# detector/__init__.py
"""Detector package"""

import logging
from datetime import datetime
from typing import Dict, Any

from .config import Config
from .database import DatabaseManager
from .camera_processor import CameraProcessor
from .models import load_models
from .frame_cache import get_latest_frame, set_latest_frame, clear_frame

logger = logging.getLogger(__name__)

# Track active cameras
_active: Dict[str, CameraProcessor] = {}

__all__ = [
    "run_camera",
    "stop_camera",
    "list_cameras",
    "stop_all",
    "get_latest_frame",
    "set_latest_frame",
    "_active"
]

logger.info("✅ Detector package initialized")


def list_cameras():
    """List all cameras from Supabase"""
    try:
        db = DatabaseManager()
        cameras = db.get_cameras()
        logger.info(f"Loaded {len(cameras)} cameras")
        return cameras
    except Exception as e:
        logger.error(f"Error listing cameras: {e}")
        return []


def run_camera(camera_id: str) -> Dict[str, Any]:
    """Start emotion detection for a camera"""
    try:
        # Check if already running
        if camera_id in _active:
            return {
                "status": "already_running",
                "camera": camera_id,
                "time": datetime.now().isoformat()
            }

        # Load camera from database
        db = DatabaseManager()
        cameras = db.get_cameras()
        cam = next((c for c in cameras if c["id"] == camera_id), None)

        if not cam:
            raise ValueError(f"Camera {camera_id} not found")

        if not cam.get("enabled", True):
            raise ValueError(f"Camera {camera_id} is disabled")

        # Load models
        models = load_models()

        # Create and start processor
        proc = CameraProcessor(cam, models, db)
        ok = proc.start_processing()

        if not ok:
            raise RuntimeError(f"Failed to start camera {camera_id}")

        _active[camera_id] = proc

        logger.info(f"✅ Camera {camera_id} started")
        return {
            "status": "started",
            "camera": camera_id,
            "name": cam.get("name", camera_id),
            "location": cam.get("location_zone", "unknown"),
            "time": datetime.now().isoformat()
        }

    except ValueError:
        raise
    except RuntimeError:
        raise
    except Exception as e:
        logger.error(f"Unexpected error starting {camera_id}: {e}")
        raise RuntimeError(f"Failed to start camera: {str(e)}")


def stop_camera(camera_id: str) -> Dict[str, Any]:
    """Stop a running camera"""
    if camera_id not in _active:
        raise ValueError(f"Camera {camera_id} is not running")

    try:
        proc = _active[camera_id]
        proc.stop_processing()
        del _active[camera_id]

        logger.info(f"✅ Camera {camera_id} stopped")
        return {
            "status": "stopped",
            "camera": camera_id,
            "time": datetime.now().isoformat()
        }

    except Exception as e:
        logger.error(f"Error stopping {camera_id}: {e}")
        raise


def stop_all() -> Dict[str, Any]:
    """Stop all running cameras"""
    camera_ids = list(_active.keys())
    stopped = []
    errors = []

    for camera_id in camera_ids:
        try:
            stop_camera(camera_id)
            stopped.append(camera_id)
        except Exception as e:
            errors.append({"camera": camera_id, "error": str(e)})

    logger.info(f"✅ Stopped {len(stopped)} cameras")

    result = {
        "status": "all_stopped",
        "stopped": stopped,
        "time": datetime.now().isoformat()
    }

    if errors:
        result["errors"] = errors

    return result