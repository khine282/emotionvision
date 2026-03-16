# main.py
import logging
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from services.detector_service import router as detector_router
from detector.config import Config

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)

Config.validate()

app = FastAPI(
    title="EmotionVision API",
    version="2.0.0",
    description="Real-time emotion detection API"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# API Routes
app.include_router(detector_router, prefix="/api/detector", tags=["Detector"])

@app.get("/health", tags=["Health"])
def health_check():
    return {"status": "healthy"}

# Serve frontend (must be last!)
app.mount("/", StaticFiles(directory="static", html=True), name="static")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host=Config.HOST, port=Config.PORT)