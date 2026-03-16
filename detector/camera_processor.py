# detector/camera_processor.py
"""Camera processing - captures frames and runs emotion detection"""

import cv2
import numpy as np
import threading
import logging
import time
from datetime import datetime
from typing import Dict, Any, Optional

from detector.config import Config
from detector.frame_cache import set_latest_frame, clear_frame
from detector.database import DatabaseManager

logger = logging.getLogger(__name__)

class CameraProcessor:
    """Handles frame capture and emotion detection for a single camera"""

    def __init__(self, camera_config: Dict, models: Dict, db: DatabaseManager):
        self.camera_config = camera_config
        self.models = models
        self.db = db

        self.camera_id = camera_config["id"]
        self.camera_name = camera_config["name"]
        self.source = camera_config["source"]
        self.camera_type = camera_config["type"]

        self._capture_thread = None
        self._process_thread = None
        self._stop_event = threading.Event()
        self._cap = None

        self._frame_buffer = None
        self._buffer_lock = threading.Lock()

        self._fps = 0
        self._frame_count = 0
        self._last_fps_time = time.time()

        logger.info(f"Camera processor initialized: {self.camera_name} ({self.camera_id})")

    def start_processing(self) -> bool:
        """Start capture and processing threads"""
        try:
            # Connect to camera
            if not self._connect_camera():
                return False

            self._stop_event.clear()

            # Start capture thread
            self._capture_thread = threading.Thread(
                target=self._capture_loop,
                daemon=True,
                name=f"capture_{self.camera_id}"
            )
            self._capture_thread.start()

            # Start processing thread
            self._process_thread = threading.Thread(
                target=self._process_loop,
                daemon=True,
                name=f"process_{self.camera_id}"
            )
            self._process_thread.start()

            logger.info(f"✅ [{self.camera_id}] Processing started")
            return True

        except Exception as e:
            logger.error(f"[{self.camera_id}] Failed to start: {e}")
            return False

    def stop_processing(self):
        """Stop all threads and release resources"""
        logger.info(f"[{self.camera_id}] Stopping...")
        self._stop_event.set()

        if self._capture_thread:
            self._capture_thread.join(timeout=5)

        if self._process_thread:
            self._process_thread.join(timeout=5)

        if self._cap:
            self._cap.release()
            self._cap = None

        clear_frame(self.camera_id)
        logger.info(f"[{self.camera_id}] Stopped and cleaned up")

    def _connect_camera(self) -> bool:
        """Connect to camera source"""
        try:
            if self.camera_type == "webcam":
                source = int(self.source) if str(self.source).isdigit() else 0
            else:
                source = self.source

            self._cap = cv2.VideoCapture(source)

            if not self._cap.isOpened():
                logger.error(f"[{self.camera_id}] Failed to open camera: {source}")
                return False

            self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, Config.FRAME_WIDTH)
            self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, Config.FRAME_HEIGHT)
            self._cap.set(cv2.CAP_PROP_FPS, Config.TARGET_FPS)

            logger.info(f"[{self.camera_id}] Connected to camera: {source}")
            return True

        except Exception as e:
            logger.error(f"[{self.camera_id}] Connection error: {e}")
            return False

    def _capture_loop(self):
        """Continuously capture frames from camera"""
        logger.info(f"[{self.camera_id}] Capture loop started")

        while not self._stop_event.is_set():
            try:
                if self._cap is None or not self._cap.isOpened():
                    logger.warning(f"[{self.camera_id}] Camera disconnected, retrying...")
                    time.sleep(2)
                    self._connect_camera()
                    continue

                ret, frame = self._cap.read()

                if not ret or frame is None:
                    time.sleep(0.033)
                    continue

                with self._buffer_lock:
                    self._frame_buffer = frame.copy()

            except Exception as e:
                logger.error(f"[{self.camera_id}] Capture error: {e}")
                time.sleep(0.1)

        logger.info(f"[{self.camera_id}] Capture loop stopped")

    def _process_loop(self):
        """Process frames - run face detection and emotion recognition"""
        logger.info(f"[{self.camera_id}] Processing loop started")
        last_screenshot_time = 0

        while not self._stop_event.is_set():
            try:
                # Get latest frame
                with self._buffer_lock:
                    if self._frame_buffer is None:
                        time.sleep(0.033)
                        continue
                    frame = self._frame_buffer.copy()

                # Run detection
                processed_frame, detections = self._detect_emotions(frame)

                # Update frame cache for streaming
                set_latest_frame(self.camera_id, processed_frame)

                # Save detections to database
                for detection in detections:
                    self.db.insert_emotion_detection(
                        camera_id=self.camera_id,
                        face_id=detection["face_id"],
                        emotion=detection["emotion"],
                        confidence=detection["confidence"],
                    )

                    # Auto screenshot if enabled
                    now = time.time()
                    if (Config.AUTO_SCREENSHOT and
                        detection["confidence"] >= Config.SCREENSHOT_CONFIDENCE_THRESHOLD and
                        now - last_screenshot_time >= Config.MIN_SCREENSHOT_INTERVAL):

                        self._take_screenshot(processed_frame, detection)
                        last_screenshot_time = now

                # FPS tracking
                self._frame_count += 1
                now = time.time()
                if now - self._last_fps_time >= 30:
                    self._fps = self._frame_count / (now - self._last_fps_time)
                    logger.info(f"[{self.camera_id}] FPS: {self._fps:.1f}")
                    self._frame_count = 0
                    self._last_fps_time = now

                time.sleep(0.033)  # ~30 FPS

            except Exception as e:
                logger.error(f"[{self.camera_id}] Processing error: {e}")
                time.sleep(0.1)

        logger.info(f"[{self.camera_id}] Processing loop stopped")

    def _detect_emotions(self, frame: np.ndarray):
        """Run face detection and emotion recognition on a frame"""
        detections = []
        display_frame = frame.copy()

        try:
            face_detector = self.models.get("face_detection")
            emotion_model = self.models.get("emotion_model")

            if face_detector is None:
                return display_frame, detections


            # Convert to RGB for MediaPipe
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_detector.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            ) 
            for i, (x, y, w, h) in enumerate(faces):
                try: 
                    emotion = "neutral"
                    confidence = 0.0

                    # Run emotion model
                    if emotion_model is not None:
                        input_size = self.models.get("input_size", (64, 64))
                        face_roi = gray[y:y+h, x:x+w]  # ← fixed
                        resized = cv2.resize(face_roi, input_size)
                        normalized = resized / 255.0
                        input_tensor = normalized.reshape(1, input_size[0], input_size[1], 1)

                        predictions = emotion_model.predict(input_tensor, verbose=0)
                        emotion_idx = np.argmax(predictions[0])
                        confidence = float(predictions[0][emotion_idx])
                        emotion = self.models["emotion_classes"].get(emotion_idx, "neutral")

                    face_id = f"{self.camera_id}_face_{i}"

                    detections.append({
                        "face_id": face_id,
                        "emotion": emotion,
                        "confidence": confidence,
                        "bbox": (x, y, w, h)
                    })

                    # Draw on frame
                    color = (0, 255, 0) if confidence > 0.6 else (0, 165, 255)
                    cv2.rectangle(display_frame, (x, y), (x+w, y+h), color, 2)
                    label = f"{emotion} {confidence:.0%}"
                    cv2.putText(display_frame, label, (x, y-10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

                except Exception as e:
                    logger.error(f"Face processing error: {e}")
                    continue

        except Exception as e:
            logger.error(f"Detection error: {e}")

        return display_frame, detections

    def _take_screenshot(self, frame: np.ndarray, detection: Dict):
        """Encode frame and upload to Supabase Storage"""
        try:
            timestamp = datetime.now()
            timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S_%f")[:-3]
            filename = f"{self.camera_id}_{detection['emotion']}_{timestamp_str}.jpg"

            # Encode frame to bytes
            ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if not ret:
                return

            file_bytes = buffer.tobytes()

            # Upload to Supabase Storage
            public_url, storage_path = self.db.upload_screenshot(file_bytes, filename)

            if public_url:
                self.db.insert_screenshot(
                    camera_id=self.camera_id,
                    face_id=detection["face_id"],
                    emotion=detection["emotion"],
                    confidence=detection["confidence"],
                    filename=filename,
                    storage_path=storage_path,
                    public_url=public_url,
                    file_size_bytes=len(file_bytes),
                    trigger_reason="auto_detection"
                )

        except Exception as e:
            logger.error(f"Screenshot error: {e}")

    @property
    def fps(self):
        return self._fps