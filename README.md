# Real-time WebRTC VLM: Multi-Object Detection

[![Demo Video](https://www.loom.com/share/cba83f3feb024f2bb8dd52bcf1dfdcf3?sid=3dad1f71-8baa-4756-91f3-39aff693267e)](https://www.loom.com/share/cba83f3feb024f2bb8dd52bcf1dfdcf3?sid=3dad1f71-8baa-4756-91f3-39aff693267e)

---

## Project Introduction

**WebRTC VLM** is a real-time, phone-to-browser video streaming application with multi-object detection. It supports two inference modes:
- **Server Mode**: Video frames are sent to a Python backend for object detection using a YOLOv5 ONNX model.
- **WASM Mode**: The same ONNX model runs directly in the browser using WebAssembly for a serverless, low-resource alternative.

**Key Features:**
- Real-time, low-latency video streaming from phone to laptop using WebRTC.
- Dual inference modes (server and WASM) for flexibility and performance.
- Live bounding box overlays for detected objects.
- One-way communication: phone as video source, laptop as viewer/processor.
- Secure tunneling for mobile testing using `serveo.net` and `ngrok`.

**Target Audience:**
- Developers, researchers, and students interested in real-time computer vision, WebRTC, and ONNX model deployment in both server and browser environments.

---

## Tech Stack

### Frontend
- **Next.js**: 15.4.7
- **React**: 19.1.0
- **TypeScript**
- **Tailwind CSS**
- **Socket.IO Client**: ^4.8.1
- **onnxruntime-web**: ^1.22.0
- **qrcode.react**: ^4.2.0

### Backend (Signaling)
- **Node.js**: 18+
- **Express**: ^5.1.0
- **Socket.IO**: ^4.8.1
- **nodemon**: ^3.1.10

### Backend (Inference)
- **Python**: 3.9+
- **FastAPI**
- **Uvicorn**
- **onnxruntime**
- **NumPy**
- **Pillow**
- **python-multipart**

### Tooling
- **uv** (Python environment manager)
- **ngrok** (tunneling)
- **SSH** (for serveo.net tunnels)

---

## High-Level Design

The project consists of three main components:

1.  **Frontend (Next.js/React)**
    - Provides the user interface for both phone (video source) and laptop (viewer/detector).
    - Handles WebRTC peer connection setup, QR code generation, and displays detection overlays.
    - Supports both server and WASM inference modes.

2.  **Signaling Server (Node.js/Express/Socket.IO)**
    - Facilitates WebRTC peer discovery and connection setup between phone and laptop.
    - Relays signaling messages (offer, answer, ICE candidates, mode selection).

3.  **Inference Server (Python/FastAPI)**
    - Receives video frames from the frontend in server mode.
    - Runs YOLOv5 ONNX model inference and returns detection results.

**Component Interaction:**
- The phone streams video to the laptop via WebRTC (signaling handled by the Node.js server).
- In server mode, the laptop sends video frames to the Python inference server for detection.
- In WASM mode, detection runs directly in the browser using onnxruntime-web.
- Detection results are rendered as overlays on the video in real-time.

---

## Low-Level Design

### Key Modules & Files

#### Frontend
- `frontend/src/app/page.tsx`: Main React component. Handles WebRTC logic, QR code generation, mode switching, and detection overlay rendering.
- `frontend/src/lib/inference.ts`: Implements YOLOv5 inference in the browser using onnxruntime-web. Handles image preprocessing, model loading, and postprocessing.
- `frontend/public/yolov5n.onnx`: ONNX model for WASM mode.

#### Signaling Server
- `server/index.js`: Express + Socket.IO server. Handles signaling events, room management, and relays WebRTC and mode selection messages.

#### Inference Server
- `server/main.py`: FastAPI app exposing `/detect` endpoint for object detection. Handles CORS for frontend communication.
- `server/inference.py`: Loads YOLOv5 ONNX model, preprocesses images, runs inference, and postprocesses results (including NMS).
- `server/yolov5n.onnx`: ONNX model for server mode.

### Design Patterns & Algorithms
- **Singleton**: The Python inference server uses a single model instance for efficiency.
- **Non-Maximum Suppression (NMS)**: Used in server-side postprocessing to filter overlapping bounding boxes.
- **WebRTC Signaling**: Custom room-based signaling for peer connection setup.
- **Modularization**: Clear separation of frontend, signaling, and inference logic.

---

## How to Run the Project Locally

### Prerequisites
- **Git**
- **Node.js**: v18 or higher
- **Python**: v3.9 or higher
- **uv**: Python package manager (`pip install uv`)
- **SSH Client**
- **ngrok**: [Download here](https://ngrok.com/download) and set up your authtoken

### Step-by-Step Instructions

#### 1. Clone the Repository
```bash
# Clone the repo and enter the directory
git clone [https://github.com/devvratsinghpanwar/webrtc-vlm](https://github.com/devvratsinghpanwar/webrtc-vlm)
cd webrtc-vlm
```

#### 2. Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

#### 3. Install Signaling Server Dependencies
```bash
cd server
npm install
cd ..
```

#### 4. Set Up Python Environment for Inference Server
```bash
cd server
uv venv

# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1

# On macOS/Linux:
# source .venv/bin/activate

uv pip install -r requirements.txt
cd ..
```

#### 5. Start All Services (Use 6 Terminals)
- **Terminal 1: Start signaling server**
  ```bash
  cd server
  npm run dev
  # Runs at http://localhost:4000
  ```
- **Terminal 2: Start inference server**
  ```bash
  cd server
  # Make sure your venv is active!
  uvicorn main:app --reload
  # Runs at http://localhost:8000
  ```
- **Terminal 3: Start frontend**
  ```bash
  cd frontend
  npm run dev
  # Runs at http://localhost:3000
  ```
- **Terminal 4: Expose frontend via serveo**
  ```bash
  ssh -R 80:localhost:3000 serveo.net
  # Copy the public URL (LAPTOP_URL)
  ```
- **Terminal 5: Expose signaling server via serveo**
  ```bash
  ssh -R 80:localhost:4000 serveo.net
  # Copy the public URL (SIGNALING_SERVER_URL)
  ```
- **Terminal 6: Expose inference server via ngrok**
  ```bash
  ngrok http 8000
  # Copy the public URL (INFERENCE_SERVER_URL)
  ```

#### 6. Configure URLs in Frontend
Edit `frontend/src/app/page.tsx`:
```javascript
// Tunnel to port 4000 (from Terminal 5)
const SIGNALING_SERVER_URL = "[https://something2.serveo.net](https://something2.serveo.net)";
// Tunnel to port 3000 (from Terminal 4)
const LAPTOP_URL = "[https://something1.serveo.net](https://something1.serveo.net)";
// Tunnel to port 8000 (from Terminal 6)
const INFERENCE_SERVER_URL = "[https://something3.ngrok-free.app](https://something3.ngrok-free.app)";
```
Save the file. The Next.js dev server will reload automatically.

#### 7. Run the Application!
1. On your **laptop**, open your browser and go to `http://localhost:3000`.
2. You will see two QR codes.
3. On your **phone**, scan the "Scan for Server Mode" or "Scan for WASM Mode" QR code.
4. Grant camera permissions when prompted.
5. The live video stream with object detection overlays will appear on your laptop!

---

## Project Structure
```
├── frontend/
│   ├── public/
│   │   └── yolov5n.onnx          # Model for WASM mode
│   ├── src/
│   │   ├── page.tsx              # Main application component
│   │   └── inference.ts          # Client-side WASM inference logic
│   └── package.json
├── server/
│   ├── .venv/                    # Python virtual environment
│   ├── index.js                  # Node.js Signaling Server
│   ├── main.py                   # Python Inference Server (FastAPI)
│   ├── inference.py              # Server-side model logic
│   ├── yolov5n.onnx              # Model for Server mode
│   ├── package.json
│   └── requirements.txt
└── README.md
```

---

## Additional Information

### Troubleshooting
- If you see CORS errors, ensure the correct URLs are whitelisted in `server/main.py`.
- If the WASM model fails to load, check that all ONNX Runtime WASM files are present in `frontend/public/static/wasm/`.
- If ngrok or serveo tunnels disconnect, restart the tunnel and update the URLs in `page.tsx`.

### Testing
No automated tests are included, but you can verify functionality by running the full stack and checking for live detections.

### Extending
- You can swap the ONNX model for a different one (ensure input/output shapes match YOLOv5n).
- Add more UI features or support for two-way video/audio as needed.

### License
This project is for educational and research purposes. See individual file headers for more details.
