# Real-time WebRTC VLM: Multi-Object Detection

This project demonstrates a real-time, phone-to-browser video streaming application that performs multi-object detection and overlays the results on the live feed. It is designed to run in two modes: a high-performance **Server Mode** and a low-resource **WASM Mode**.

![Demo GIF]([https://i.imgur.com/placeholder.gif](https://www.loom.com/share/cba83f3feb024f2bb8dd52bcf1dfdcf3?sid=3dad1f71-8baa-4756-91f3-39aff693267e)) 

---

## Features

-   **Real-time Video Streaming**: Low-latency video from a phone's camera to a laptop browser using WebRTC.
-   **Dual Inference Modes**:
    -   **Server Mode**: Video frames are sent to a Python backend for powerful, server-side object detection using an ONNX model.
    -   **WASM Mode**: The same ONNX model runs directly in the browser using WebAssembly for a serverless, low-resource alternative.
-   **Live Bounding Box Overlays**: Detection results are rendered on the video in real-time.
-   **One-Way Communication**: The phone acts as a video source, and the laptop as the viewer and processor.
-   **Secure Tunneling**: Uses `serveo.net` and `ngrok` to securely expose local servers for mobile testing.

---

## Technology Stack

-   **Frontend**: Next.js, React, TypeScript, Tailwind CSS, Socket.IO Client, ONNX Runtime Web (for WASM)
-   **Backend (Inference)**: Python, FastAPI, ONNX Runtime, NumPy, Pillow
-   **Backend (Signaling)**: Node.js, Express, Socket.IO
-   **Tooling**: `uv` (Python environment), `nodemon`, SSH, `ngrok`

---

## Local Development Setup (Manual Guide)

This guide explains how to run the entire application manually across six terminals without using Docker.

### Prerequisites

Before you begin, ensure you have the following installed:

-   **Git**: For cloning the repository.
-   **Node.js**: Version 18 or higher.
-   **Python**: Version 3.9 or higher.
-   **`uv`**: The fast Python package manager. If you don't have it, run: `pip install uv`.
-   **SSH Client**: Included by default on Windows (PowerShell/CMD), macOS, and Linux.
-   **`ngrok`**: A tunneling tool. [Install it from the official website](https://ngrok.com/download) and add your authtoken (you only need a free account).

### Step-by-Step Instructions

#### 1. Clone the Repository

```bash
git clone https://github.com/devvratsinghpanwar/webrtc-vlm
cd webrtc-vlm-project

cd frontend
npm install
cd .. 

cd server

# Install Node.js dependencies for the signaling server
npm install

# Create and activate a Python virtual environment using uv
uv venv
# On Windows (PowerShell):
.\.venv\Scripts\Activate.ps1
# On macOS/Linux:
# source .venv/bin/activate

# Install Python dependencies for the inference server
uv pip install -r requirements.txt

cd .. 
# Return to the project root

#now you will need 6 different terminals
# 1st terminal
cd server
npm run dev
# This will run on http://localhost:4000

#2nd terminal
cd server
# Make sure your venv is active!
uvicorn main:app --reload
# This will run on http://localhost:8000

# 3rd terminal
cd frontend
npm run dev
# This will run on http://localhost:3000

# now expose frontend via servio
# 4th terminal
ssh -R 80:localhost:3000 serveo.net
# It will give you a public URL like: https://something1.serveo.net
# Copy this URL. This is your LAPTOP_URL. to be replace in page.tsx

# 5th terminal
# expose signalling server via serveo
ssh -R 80:localhost:4000 serveo.net
# It will give you a public URL like: https://something2.serveo.net
# Copy this URL. This is your SIGNALING_SERVER_URL. to be replaced in page.tsx

#expose inference server the object detection one
# 6th terminal
ngrok http 8000
# It will give you a public URL like: https://something3.ngrok-free.app
# Copy this URL. This is your INFERENCE_SERVER_URL. to be replaced in page.tsx
```

## this is how page.tsx should look like
// frontend/src/app/page.tsx

// Tunnel to port 4000 (from Terminal 5)
const SIGNALING_SERVER_URL = "https://something2.serveo.net";

// Tunnel to port 3000 (from Terminal 4)
const LAPTOP_URL = "https://something1.serveo.net";

// Tunnel to port 8000 (from Terminal 6)
const INFERENCE_SERVER_URL = "https://something3.ngrok-free.app";```

Save the file. The Next.js development server in Terminal 3 will automatically reload.

#### 6. Run the Application!

1.  On your **laptop**, open your browser and navigate to **http://localhost:3000**.
2.  You will see two QR codes.
3.  On your **phone**, scan the "Scan for Server Mode" or "Scan for WASM Mode" QR code.
4.  Grant camera permissions when your phone's browser asks.
5.  The live video stream with object detection overlays will now appear on your laptop!

## Project Structure
```bash
├── frontend/
│ ├── public/
│ │ └── yolov5n.onnx # Model for WASM mode
│ ├── src/
│ │ ├── app/page.tsx # Main application component
│ │ └── lib/inference.ts # Client-side WASM inference logic
│ └── package.json
├── server/
│ ├── .venv/ # Python virtual environment
│ ├── index.js # Node.js Signaling Server
│ ├── main.py # Python Inference Server (FastAPI)
│ ├── inference.py # Server-side model logic
│ ├── yolov5n.onnx # Model for Server mode
│ ├── package.json
│ └── requirements.txt
└── README.md
```
