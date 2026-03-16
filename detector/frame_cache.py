# detector/frame_cache.py
"""In-memory frame cache for active cameras"""

import threading
from typing import Dict, Optional
import numpy as np

_frame_cache: Dict[str, np.ndarray] = {}
_lock = threading.Lock()

def set_latest_frame(camera_id: str, frame: np.ndarray):
    """Store latest frame for a camera"""
    with _lock:
        _frame_cache[camera_id] = frame.copy()

def get_latest_frame(camera_id: str) -> Optional[np.ndarray]:
    """Get latest frame for a camera"""
    with _lock:
        return _frame_cache.get(camera_id)

def clear_frame(camera_id: str):
    """Remove frame from cache when camera stops"""
    with _lock:
        if camera_id in _frame_cache:
            del _frame_cache[camera_id]
            
def clear_all_frames():
    """Clear all frames from cache"""
    with _lock:
        _frame_cache.clear()