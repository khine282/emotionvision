# detector/config.py
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Supabase
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_ANON_KEY = os.getenv("SUPABASE_ANON_KEY")
    SUPABASE_SERVICE_KEY = os.getenv("SUPABASE_SERVICE_KEY")

    # Detection settings
    FRAME_WIDTH = 640
    FRAME_HEIGHT = 480
    TARGET_FPS = 15
    CONFIDENCE_THRESHOLD = 0.5

    # Screenshot settings
    SCREENSHOT_BUCKET = "screenshots"
    AUTO_SCREENSHOT = True
    SCREENSHOT_CONFIDENCE_THRESHOLD = 0.75
    MIN_SCREENSHOT_INTERVAL = 10.0

    # Server
    HOST = os.getenv("HOST", "0.0.0.0")
    PORT = int(os.getenv("PORT", 8000))

    @classmethod
    def validate(cls):
        """Check all required env vars are set"""
        missing = []
        for key in ["SUPABASE_URL", "SUPABASE_ANON_KEY", "SUPABASE_SERVICE_KEY"]:
            if not getattr(cls, key):
                missing.append(key)
        if missing:
            raise ValueError(f"Missing environment variables: {missing}")