# detector/database.py
"""Database operations using Supabase"""

import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
from supabase import create_client, Client
from detector.config import Config

logger = logging.getLogger(__name__)

class DatabaseManager:
    """Handle all database operations via Supabase"""

    def __init__(self):
        """Initialize Supabase client"""
        self.client: Client = create_client(
            Config.SUPABASE_URL,
            Config.SUPABASE_SERVICE_KEY
        )
        logger.info("✅ Connected to Supabase")

    # ============================================================================
    # Camera Operations
    # ============================================================================

    def get_cameras(self) -> List[Dict]:
        """Get all enabled cameras"""
        try:
            result = self.client.table("cameras")\
                .select("*")\
                .eq("is_enabled", True)\
                .execute()

            cameras = []
            for row in result.data:
                cameras.append({
                    "id": row["camera_id"],
                    "name": row["camera_name"],
                    "type": row["camera_type"],
                    "source": row["source"],
                    "location_zone": row["location_zone"],
                    "enabled": row["is_enabled"],
                    "output_dir": f"output_{row['location_zone']}",
                })
            return cameras

        except Exception as e:
            logger.error(f"Error getting cameras: {e}")
            return []

    def add_camera(self, camera_id: str, camera_name: str, camera_type: str,
                   source: str, location_zone: str, is_enabled: bool = True) -> bool:
        """Add a new camera"""
        try:
            self.client.table("cameras").upsert({
                "camera_id": camera_id,
                "camera_name": camera_name,
                "camera_type": camera_type,
                "source": source,
                "location_zone": location_zone,
                "is_enabled": is_enabled,
            }).execute()
            logger.info(f"✅ Camera added: {camera_id}")
            return True

        except Exception as e:
            logger.error(f"Error adding camera: {e}")
            return False

    def update_camera(self, camera_id: str, updates: Dict) -> bool:
        """Update camera fields"""
        try:
            self.client.table("cameras")\
                .update(updates)\
                .eq("camera_id", camera_id)\
                .execute()
            return True

        except Exception as e:
            logger.error(f"Error updating camera: {e}")
            return False

    def delete_camera(self, camera_id: str) -> bool:
        """Soft delete — disable camera"""
        try:
            self.client.table("cameras")\
                .update({"is_enabled": False})\
                .eq("camera_id", camera_id)\
                .execute()
            return True

        except Exception as e:
            logger.error(f"Error deleting camera: {e}")
            return False

    # ============================================================================
    # Emotion Detection Operations
    # ============================================================================

    def insert_emotion_detection(self, camera_id: str, face_id: str,
                                  emotion: str, confidence: float,
                                  event_type: str = "detection",
                                  stable_duration: float = None,
                                  previous_emotion: str = None,
                                  processing_time_ms: float = None) -> bool:
        """Insert emotion detection record"""
        try:
            self.client.table("emotion_detections").insert({
                "camera_id": camera_id,
                "face_id": face_id,
                "emotion": emotion,
                "confidence": confidence,
                "event_type": event_type,
                "stable_duration": stable_duration,
                "previous_emotion": previous_emotion,
                "processing_time_ms": processing_time_ms,
                "timestamp": datetime.now().isoformat(),
            }).execute()
            return True

        except Exception as e:
            logger.error(f"Error inserting emotion detection: {e}")
            return False

    # ============================================================================
    # Screenshot Operations
    # ============================================================================

    def insert_screenshot(self, camera_id: str, face_id: str,
                          emotion: str, confidence: float,
                          filename: str, storage_path: str,
                          public_url: str, file_size_bytes: int,
                          trigger_reason: str) -> bool:
        """Insert screenshot record"""
        try:
            self.client.table("screenshots").insert({
                "camera_id": camera_id,
                "face_id": face_id,
                "emotion": emotion,
                "confidence": confidence,
                "filename": filename,
                "storage_path": storage_path,
                "public_url": public_url,
                "file_size_bytes": file_size_bytes,
                "trigger_reason": trigger_reason,
                "timestamp": datetime.now().isoformat(),
            }).execute()
            return True

        except Exception as e:
            logger.error(f"Error inserting screenshot: {e}")
            return False

    def get_screenshots(self, camera_id: str = None,
                        limit: int = 50, offset: int = 0) -> List[Dict]:
        """Get screenshots with optional camera filter"""
        try:
            query = self.client.table("screenshots")\
                .select("*")\
                .order("timestamp", desc=True)\
                .limit(limit)\
                .offset(offset)

            if camera_id:
                query = query.eq("camera_id", camera_id)

            result = query.execute()
            return result.data or []

        except Exception as e:
            logger.error(f"Error getting screenshots: {e}")
            return []

    def get_screenshot_by_id(self, screenshot_id: str) -> Optional[Dict]:
        """Get single screenshot by ID"""
        try:
            result = self.client.table("screenshots")\
                .select("*")\
                .eq("id", screenshot_id)\
                .single()\
                .execute()
            return result.data

        except Exception as e:
            logger.error(f"Error getting screenshot {screenshot_id}: {e}")
            return None

    def delete_screenshot(self, screenshot_id: str) -> Optional[Dict]:
        """Delete screenshot record, return info for file deletion"""
        try:
            # Get first so we can delete from storage too
            screenshot = self.get_screenshot_by_id(screenshot_id)
            if not screenshot:
                return None

            self.client.table("screenshots")\
                .delete()\
                .eq("id", screenshot_id)\
                .execute()

            return screenshot

        except Exception as e:
            logger.error(f"Error deleting screenshot: {e}")
            return None

    def get_screenshot_count(self, camera_id: str = None) -> int:
        """Get total screenshot count"""
        try:
            query = self.client.table("screenshots").select("id", count="exact")
            if camera_id:
                query = query.eq("camera_id", camera_id)
            result = query.execute()
            return result.count or 0

        except Exception as e:
            logger.error(f"Error getting screenshot count: {e}")
            return 0

    # ============================================================================
    # Storage Operations
    # ============================================================================

    def upload_screenshot(self, file_bytes: bytes,
                          filename: str) -> Optional[str]:
        """Upload screenshot to Supabase Storage, return public URL"""
        try:
            storage_path = f"screenshots/{filename}"

            self.client.storage.from_(Config.SCREENSHOT_BUCKET).upload(
                path=storage_path,
                file=file_bytes,
                file_options={"content-type": "image/jpeg"}
            )

            # Get public URL
            public_url = self.client.storage.from_(
                Config.SCREENSHOT_BUCKET
            ).get_public_url(storage_path)

            logger.info(f"✅ Screenshot uploaded: {filename}")
            return public_url, storage_path

        except Exception as e:
            logger.error(f"Error uploading screenshot: {e}")
            return None, None

    def delete_screenshot_file(self, storage_path: str) -> bool:
        """Delete screenshot file from Supabase Storage"""
        try:
            self.client.storage.from_(
                Config.SCREENSHOT_BUCKET
            ).remove([storage_path])
            return True

        except Exception as e:
            logger.error(f"Error deleting screenshot file: {e}")
            return False