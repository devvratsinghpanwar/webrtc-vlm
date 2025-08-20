# server/main.py
from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
import time

# Import the model instance from our inference script
from inference import model

# Create the FastAPI app
app = FastAPI()

# --- Add CORS middleware ---
# This is crucial for allowing the frontend to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "*",  # For development, allow all. For production, restrict to your domains below.
        "http://localhost:3000",
        "https://8c38f8dca28bffb4db090ed42c2f804c.serveo.net",
        "https://b0ea177fd99a8bd1b271d7030a31370f.serveo.net",
        "https://8c38f8dca28bffb4db090ed42c2f804c.serveo.net",
        "https://5e9d97509e4079f49bb08d61163d4b3b.serveo.net"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def read_root():
    """ A simple endpoint to confirm the server is running. """
    return {"status": "ok", "message": "Inference server is running!"}


@app.post("/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    The main object detection endpoint.
    Receives an image file, runs detection, and returns the results.
    """
    # Timestamps for metrics, as required by the project spec
    recv_ts = time.time() * 1000  # a more realistic timestamp in ms

    # Read the image file bytes
    image_bytes = await file.read()
    
    # Run detection
    detections = model.detect(image_bytes)
    
    inference_ts = time.time() * 1000

    # Return the data in the format specified by the UX / API contract
    return {
        "recv_ts": recv_ts,
        "inference_ts": inference_ts,
        "detections": detections,
    }