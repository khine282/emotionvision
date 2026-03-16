cd "C:\D_Drive\Uni 2026\EmotionVision"

# Create new venv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Run server
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

http://localhost:8000/docs#/Detector/list_screenshots_api_detector_screenshots_get



# For Live streaming vs local streaming

// Change this:
const API = 'https://emotionvision-api.onrender.com';

// To this for local testing:
const API = 'http://localhost:8000';