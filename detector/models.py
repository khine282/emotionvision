# detector/models.py
"""Load emotion detection models"""

import logging
import os
import numpy as np

logger = logging.getLogger(__name__)

# Emotion classes
EMOTION_CLASSES = {
    0: 'angry',
    1: 'disgust',
    2: 'fear',
    3: 'happy',
    4: 'sad',
    5: 'surprise',
    6: 'neutral'
}

_models_cache = None

def load_models():
    """Load emotion detection models (cached after first load)"""
    global _models_cache

    if _models_cache is not None:
        return _models_cache

    models = {}

    # Load emotion model
    try:
        import tensorflow as tf
        model_path = os.path.join("trained_models", "emotion_models", "emotion_converted.h5")

        if os.path.exists(model_path):
            models["emotion_model"] = tf.keras.models.load_model(model_path, compile=False)
            models["emotion_classes"] = EMOTION_CLASSES
            models["input_size"] = (64, 64)
            logger.info(f"✅ Emotion model loaded: {model_path}")
        else:
            logger.warning(f"⚠️ Model not found at {model_path}")
            models["emotion_model"] = None

    except Exception as e:
        logger.error(f"Error loading emotion model: {e}")
        models["emotion_model"] = None

    # Load face detector (MediaPipe)
    try:
        import cv2
        face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        models["face_detection"] = face_cascade
        models["detector_type"] = "haar"
        logger.info("✅ Haar Cascade face detector loaded")

    except Exception as e:
        logger.error(f"Error loading face detector: {e}")
        models["face_detection"] = None


    _models_cache = models
    return models