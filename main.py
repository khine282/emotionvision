# main.py
"""
EmotionVision API
-----------------
FastAPI backend for emotion detection
"""

import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from services.detector_service import router as detector_router
from detector.config import Config

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

# Validate env vars on startup
Config.validate()

app = FastAPI(
    title="EmotionVision API",
    version="2.0.0",
    description="Real-time emotion detection API"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(detector_router, prefix="/api/detector", tags=["Detector"])

@app.get("/", tags=["Root"])
def root():
    return {
        "message": "EmotionVision API ✅",
        "version": "2.0.0",
        "docs": "/docs"
    }

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)