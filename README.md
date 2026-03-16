# EmotionVision 👁

> Real-time emotion detection system with live camera streaming, cloud storage, and a web dashboard.

Built with FastAPI, TensorFlow, OpenCV, and Supabase — designed as a full-stack AI showcase from model inference to cloud deployment.

---

## What It Does

EmotionVision detects human emotions in real time from webcam or RTSP camera feeds. It identifies 7 emotions — **angry, disgust, fear, happy, sad, surprise, neutral** — draws bounding boxes on faces, and saves screenshots to cloud storage on demand.

![EmotionVision Dashboard](docs/dashboard.png)

---

## Features

- 🎥 **Live stream** — MJPEG stream with emotion detection overlay
- 📷 **Multi-camera support** — register webcams or RTSP IP cameras
- 😊 **7-class emotion recognition** — TensorFlow/Keras model trained on FER2013
- 📸 **Screenshot capture** — saves to Supabase Storage with public URL
- 🗄️ **Cloud database** — all camera and detection data in Supabase PostgreSQL
- 🌐 **REST API** — FastAPI with auto-generated Swagger docs
- 💻 **Web dashboard** — single-page frontend, no framework needed

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI, Python 3.11, Uvicorn |
| ML / CV | TensorFlow 2.x, OpenCV, Haar Cascade |
| Database | Supabase PostgreSQL |
| Storage | Supabase Storage |
| Frontend | HTML, CSS, Vanilla JS |
| Deploy | Render (API), Supabase (DB + Storage) |

---

## Architecture

```
Browser (Frontend)
      ↓
FastAPI (Render)
      ↓              ↓
Camera Processor   Supabase
(OpenCV + TF)    (DB + Storage)
      ↓
Frame Cache (in-memory)
      ↓
MJPEG Stream → Browser
```

---

## Project Structure

```
EmotionVision/
├── main.py                    # FastAPI app entry point
├── requirements.txt
├── render.yaml                # Render deployment config
├── static/
│   └── index.html             # Web dashboard
├── detector/
│   ├── __init__.py            # Public API (run_camera, stop_camera, etc.)
│   ├── config.py              # Environment config
│   ├── database.py            # Supabase operations
│   ├── camera_processor.py    # Frame capture + emotion detection
│   ├── models.py              # TensorFlow + Haar Cascade loader
│   └── frame_cache.py         # In-memory frame buffer
├── services/
│   └── detector_service.py    # FastAPI routes
└── trained_models/
    └── emotion_models/
        └── emotion_converted.h5
```

---

## Local Setup

### Prerequisites
- Python 3.11
- Webcam or RTSP camera
- Supabase account (free)

### 1. Clone the repo
```bash
git clone https://github.com/khine282/emotionvision.git
cd emotionvision
```

### 2. Create virtual environment
```bash
python -m venv .venv
# Windows
.\.venv\Scripts\Activate.ps1
# Mac/Linux
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up Supabase
Create a project at [supabase.com](https://supabase.com) and run this SQL:

```sql
CREATE TABLE cameras (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    camera_id TEXT UNIQUE NOT NULL,
    camera_name TEXT NOT NULL,
    camera_type TEXT NOT NULL,
    source TEXT NOT NULL,
    location_zone TEXT NOT NULL,
    is_enabled BOOLEAN DEFAULT true,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE emotion_detections (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    camera_id TEXT NOT NULL REFERENCES cameras(camera_id),
    face_id TEXT,
    emotion TEXT,
    confidence FLOAT,
    event_type TEXT DEFAULT 'detection',
    timestamp TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE screenshots (
    id UUID DEFAULT gen_random_uuid() PRIMARY KEY,
    camera_id TEXT NOT NULL REFERENCES cameras(camera_id),
    face_id TEXT,
    emotion TEXT,
    confidence FLOAT,
    filename TEXT NOT NULL,
    storage_path TEXT,
    public_url TEXT,
    file_size_bytes INTEGER,
    trigger_reason TEXT,
    timestamp TIMESTAMPTZ DEFAULT NOW()
);
```

Also create a public Storage bucket named `screenshots`.

### 5. Configure environment
Create a `.env` file:
```
SUPABASE_URL=https://your-project.supabase.co
SUPABASE_ANON_KEY=your_anon_key
SUPABASE_SERVICE_KEY=your_service_role_key
```

### 6. Add trained model
Place your trained model at:
```
trained_models/emotion_models/emotion_converted.h5
```

### 7. Run the server
```bash
uvicorn main:app --host 0.0.0.0 --port 8000 --reload
```

Open [http://localhost:8000](http://localhost:8000) for the dashboard.
Open [http://localhost:8000/docs](http://localhost:8000/docs) for the API docs.

---

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/detector/cameras` | List all cameras |
| POST | `/api/detector/cameras/add` | Register a camera |
| POST | `/api/detector/cameras/{id}/start` | Start detection |
| POST | `/api/detector/cameras/{id}/stop` | Stop detection |
| GET | `/api/detector/stream/{id}` | MJPEG live stream |
| POST | `/api/detector/cameras/{id}/screenshot` | Capture screenshot |
| GET | `/api/detector/screenshots` | List screenshots |
| DELETE | `/api/detector/screenshots/{id}` | Delete screenshot |

Full interactive docs at `/docs`.

---

## How Emotion Detection Works

```
Frame from camera
      ↓
Haar Cascade → detect face bounding boxes
      ↓
Crop face ROI → convert to grayscale → resize to 64×64
      ↓
TensorFlow model → softmax over 7 emotions
      ↓
Draw label + confidence on frame
      ↓
Store to frame cache → serve via MJPEG stream
```

---

## Deployment

The API is deployed on [Render](https://render.com) (free tier).

> ⚠️ Note: TensorFlow is not available on Render's free tier due to Python version constraints. The API, camera management, and screenshot features work fully on Render. For live emotion detection, run locally.

---

## Screenshots

| Dashboard | Live Detection | Screenshot Gallery |
|-----------|---------------|-------------------|
| ![Dashboard](docs/dashboard.png) | ![Detection](docs/detection.png) | ![Gallery](docs/gallery.png) |

---

## Author

**Kai** — Final-year Diploma in IT (Software Development), Singapore Polytechnic  
Junior AI Developer Intern @ Reachfield IT Solutions

- 🐙 GitHub: [@khine282](https://github.com/khine282)
- 💼 LinkedIn: [linkedin.com/in/yourprofile](https://linkedin.com/in/yourprofile)

---

## Related Projects

- [GlucoSG](https://github.com/khine282/glucosg) — Nutrition tracking app, 1st place SP Energized Hackathon 2025
- EmotionVision Enterprise (private) — Retail emotion analytics with RTSP, MySQL, Docker, AWS
