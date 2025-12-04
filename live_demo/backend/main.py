from fastapi import FastAPI, UploadFile, File, WebSocket, WebSocketDisconnect, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import shutil
import os
import cv2
import numpy as np
import base64
import json
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from .model_wrapper import ModelWrapper
from .video_processor import VideoProcessor
from .tunnel_manager import tunnel_manager

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Reduce ngrok logging verbosity to reduce spam
logging.getLogger("pyngrok").setLevel(logging.WARNING)

app = FastAPI(title="Urban Issue Detection API")

# CORS configuration - allow all origins for demo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Middleware to log requests (minimal logging for performance)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    # Skip logging for high-frequency endpoints to reduce overhead
    if request.url.path in ["/api/frame", "/api/latest"]:
        return await call_next(request)
    
    # Log all other requests
    import time
    start_time = time.time()
    print(f"[MIDDLEWARE] {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        duration = time.time() - start_time
        print(f"[MIDDLEWARE] Response status: {response.status_code} (took {duration:.2f}s)")
        return response
    except Exception as e:
        duration = time.time() - start_time
        print(f"[MIDDLEWARE] ERROR after {duration:.2f}s: {e}")
        import traceback
        traceback.print_exc()
        raise

# Startup event to initialize tunnels and load model
@app.on_event("startup")
async def startup_event():
    """Initialize ngrok tunnels and load model on startup"""
    logger.info("[INFO] Starting Urban Issue Detection API...")
    
    # Load model at startup to avoid delays on first request
    logger.info("[INFO] Loading detection model...")
    try:
        model = get_model()
        logger.info("[SUCCESS] Model loaded successfully!")
    except Exception as e:
        logger.error(f"[ERROR] Failed to load model: {e}")
    
    try:
        urls = tunnel_manager.start_tunnels(backend_port=8000, frontend_port=3000)
        if urls["backend_url"] and urls["frontend_url"]:
            logger.info(f"[SUCCESS] Remote access enabled!")
            logger.info(f"[INFO] Backend API: {urls['backend_url']}")
            logger.info(f"[INFO] Frontend App: {urls['frontend_url']}")
            logger.info(f"[INFO] Share the frontend URL to access from anywhere!")
        else:
            logger.info("[INFO] Running in local network mode only")
            logger.info("[INFO] Set NGROK_AUTHTOKEN in .env for remote access")
    except Exception as e:
        logger.error(f"Tunnel setup failed: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    tunnel_manager.stop_tunnels()

# Global model instances (loaded on demand or startup)
models = {}
current_model_config = {
    "type": "rcnn",
    "checkpoint": "checkpoints/best_model_rcnn_v2.pth",
    "classes": [0, 1, 3],
    "conf_threshold": 0.5
}

# Class names mapping
CLASS_NAMES = {
    0: 'Damaged Road',
    1: 'Pothole',
    2: 'Illegal Parking',
    3: 'Broken Sign',
    4: 'Fallen Tree',
    5: 'Garbage',
    6: 'Vandalism',
    7: 'Dead Animal',
    8: 'Damaged Concrete',
    9: 'Damaged Wires'
}

# Available models for switching
AVAILABLE_MODELS = {
    "rcnn": {
        "name": "Faster R-CNN",
        "checkpoint": "checkpoints/best_model_rcnn_v2.pth",
        "description": "Two-stage detector, high accuracy"
    },
    "fcos": {
        "name": "FCOS",
        "checkpoint": "checkpoints/fcos_best_model.pth",
        "description": "Anchor-free one-stage detector"
    },
    "retinanet": {
        "name": "RetinaNet",
        "checkpoint": "checkpoints/retinanet_best_model.pth",
        "description": "One-stage detector with focal loss"
    }
}

def get_model():
    key = f"{current_model_config['type']}_{current_model_config['checkpoint']}"
    if key not in models:
        # Check if checkpoint exists
        checkpoint_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../', current_model_config['checkpoint']))
        if not os.path.exists(checkpoint_path):
             # Fallback to default if specific checkpoint not found, or handle error
             print(f"Checkpoint not found: {checkpoint_path}")
        
        models[key] = ModelWrapper(
            model_type=current_model_config['type'],
            checkpoint_path=checkpoint_path,
            classes=current_model_config['classes'],
            conf_threshold=current_model_config.get('conf_threshold', 0.5)
        )
    return models[key]

def clear_model_cache():
    """Clear cached models to force reload"""
    global models
    models.clear()
    logger.info("[MODEL] Cache cleared")

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []
        self.viewers: list[WebSocket] = []

    async def connect(self, websocket: WebSocket, role: str):
        await websocket.accept()
        if role == "viewer":
            self.viewers.append(websocket)
        else:
            self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket, role: str):
        if role == "viewer":
            if websocket in self.viewers:
                self.viewers.remove(websocket)
        else:
            if websocket in self.active_connections:
                self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.viewers:
            try:
                await connection.send_json(message)
            except:
                pass

manager = ConnectionManager()

@app.get("/")
async def root():
    return {
        "message": "Urban Issue Detection API",
        "backend_url": tunnel_manager.get_backend_url(),
        "frontend_url": tunnel_manager.get_frontend_url(),
        "status": "online"
    }

@app.get("/api/info")
async def get_info():
    """Get server connection information including public URLs"""
    import socket
    
    hostname = socket.gethostname()
    try:
        # Try to find the local network IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(('10.255.255.255', 1))
            local_ip = s.getsockname()[0]
        except Exception:
            local_ip = socket.gethostbyname(hostname)
        finally:
            s.close()
    except:
        local_ip = "127.0.0.1"
    
    # Get public URLs from tunnel manager
    backend_url = tunnel_manager.get_backend_url()
    frontend_url = tunnel_manager.get_frontend_url()
    
    return {
        "local_ip": local_ip,
        "hostname": hostname,
        "backend_url": backend_url,
        "frontend_url": frontend_url,
        "public_url": backend_url,  # For backward compatibility
        "has_remote_access": frontend_url is not None,
        "qr_code_url": frontend_url if frontend_url else f"http://{local_ip}:3000"
    }

@app.get("/api/config")
async def get_config():
    """Get current model configuration"""
    return {
        "status": "ok",
        "config": current_model_config,
        "available_models": AVAILABLE_MODELS
    }

@app.post("/api/config")
async def update_config(config: dict):
    """Update model configuration and reload model"""
    global current_model_config
    
    try:
        logger.info(f"[CONFIG] Received model switch request: {config}")
        
        old_config = current_model_config.copy()
        
        # If switching model type, update checkpoint path automatically
        if "type" in config and config["type"] in AVAILABLE_MODELS:
            config["checkpoint"] = AVAILABLE_MODELS[config["type"]]["checkpoint"]
            logger.info(f"[CONFIG] Switching from {old_config.get('type')} to {config['type']}")
        
        current_model_config.update(config)
        
        # Clear model cache if model type or checkpoint changed
        if (old_config.get("type") != current_model_config.get("type") or 
            old_config.get("checkpoint") != current_model_config.get("checkpoint")):
            clear_model_cache()
            logger.info(f"[MODEL] Model cache cleared, will reload on next inference")
        
        # Broadcast model change to all WebSocket viewers
        try:
            await manager.broadcast({
                "type": "model_changed",
                "config": current_model_config
            })
            logger.info(f"[CONFIG] Broadcasted model change to {len(manager.viewers)} viewers")
        except Exception as e:
            logger.error(f"[CONFIG] Failed to broadcast: {e}")
        
        return {
            "status": "updated",
            "config": current_model_config,
            "message": f"Model switched to {current_model_config['type']}"
        }
    except Exception as e:
        logger.error(f"[CONFIG] Error updating config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/predict_video")
async def predict_video(file: UploadFile = File(...)):
    temp_dir = Path("temp_videos")
    temp_dir.mkdir(exist_ok=True)
    
    input_path = temp_dir / file.filename
    output_path = temp_dir / f"processed_{file.filename}"
    
    with open(input_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    model = get_model()
    processor = VideoProcessor(model)
    
    try:
        processor.process_video(str(input_path), str(output_path))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    return FileResponse(output_path)

@app.websocket("/api/ws/stream")
async def websocket_endpoint(websocket: WebSocket, role: str = "source"):
    await manager.connect(websocket, role)
    logger.info(f"WebSocket connected: role={role}")
    
    if role == "source":
        model = get_model()
    
    try:
        while True:
            if role == "viewer":
                # Viewers just wait for messages to keep connection alive
                try:
                    await websocket.receive_text()
                except:
                    pass
                continue
                
            # Source (Phone) logic - REAL-TIME: Only process latest frame
            try:
                # Receive first frame
                data = await websocket.receive_text()
                
                # Drain the queue - keep only the LATEST frame
                # This ensures we process the most recent frame, not old queued ones
                while True:
                    try:
                        # Try to get next frame without blocking (with very short timeout)
                        next_data = await asyncio.wait_for(
                            websocket.receive_text(), 
                            timeout=0.001  # 1ms timeout
                        )
                        data = next_data  # Use the newer frame
                        logger.debug("Dropped old frame, using newer one")
                    except asyncio.TimeoutError:
                        # No more frames in queue, use current one
                        break
                    except:
                        # Any other error, use current frame
                        break
                
                # Parse data URL format: data:image/jpeg;base64,/9j/4AAQ...
                if data.startswith('data:'):
                    # Split by comma to get the base64 part
                    if ',' in data:
                        header, encoded = data.split(',', 1)
                    else:
                        logger.warning("Invalid data URL format - no comma found")
                        continue
                else:
                    # Assume it's pure base64
                    encoded = data
                
                # Decode base64
                try:
                    image_data = base64.b64decode(encoded)
                except Exception as e:
                    logger.error(f"Base64 decode error: {e}")
                    continue
                
                if len(image_data) == 0:
                    logger.warning("Empty image data received")
                    continue
                
                # Convert to numpy array
                nparr = np.frombuffer(image_data, np.uint8)
                
                # Decode image
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.warning("Failed to decode image - frame is None")
                    continue
                
                logger.info(f"Processing latest frame: {frame.shape}")
                    
                # Convert BGR to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Inference (now includes severity classification)
                boxes, labels, scores, severities = model.predict(frame_rgb)
                
                logger.info(f"Detected {len(boxes)} objects")
                
                # Prepare result with class names and severities
                class_names_list = [CLASS_NAMES.get(label, f"Class {label}") for label in labels]
                result = {
                    "boxes": boxes,
                    "labels": labels,
                    "class_names": class_names_list,
                    "scores": scores,
                    "severities": severities,
                    "image": data # Echo back the image so viewer can display it
                }
                
                # Broadcast to viewers
                await manager.broadcast(result)
                
            except RuntimeError as e:
                # WebSocket disconnected (ngrok closes it)
                if "disconnect message has been received" in str(e):
                    logger.info(f"WebSocket closed by client (role={role})")
                    break  # Exit loop instead of continuing
                else:
                    logger.error(f"Runtime error: {e}")
                    break
            except Exception as e:
                logger.error(f"Error processing frame: {e}")
                # For other errors, continue to next frame
                continue
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket disconnected: role={role}")
        manager.disconnect(websocket, role)
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
        manager.disconnect(websocket, role)

# HTTP Polling fallback for when WebSocket doesn't work through ngrok
latest_frame_result = {"boxes": [], "labels": [], "scores": [], "image": None}
is_processing = False  # Flag to prevent concurrent processing

@app.get("/api/health")
async def health_check(request: Request):
    """Simple health check to verify backend is reachable"""
    client_ip = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")
    
    print("=" * 80)
    print(f"[HEALTH CHECK] ENDPOINT HIT!")
    print(f"[HEALTH CHECK] Client IP: {client_ip}")
    print(f"[HEALTH CHECK] User-Agent: {user_agent}")
    print(f"[HEALTH CHECK] Headers: {dict(request.headers)}")
    print("=" * 80)
    
    logger.info(f"[Health Check] Hit from {client_ip}")
    
    return {
        "status": "ok", 
        "message": "Backend is reachable", 
        "timestamp": __import__('time').time(),
        "client_ip": client_ip,
        "user_agent": user_agent
    }

@app.post("/api/frame")
async def receive_frame(request: Request):
    """HTTP fallback endpoint to receive frames from phone when WebSocket fails"""
    global latest_frame_result, is_processing
    
    logger.info("[HTTP] Frame received from phone")
    
    # Skip if already processing a frame (drop frame for real-time performance)
    if is_processing:
        logger.warning("[HTTP] Frame dropped - still processing previous frame")
        return {"status": "busy", "message": "Processing previous frame"}
    
    is_processing = True
    
    try:
        # Parse JSON
        data = await request.json()
        image_data = data.get("image")
        
        if not image_data:
            logger.error("[HTTP] No image data in request")
            return {"error": "No image data"}
        
        # Load model (cached after first load)
        model = get_model()
        
        # Process the base64 image
        if image_data.startswith('data:image'):
            if ',' in image_data:
                header, encoded = image_data.split(',', 1)
            else:
                return {"error": "Invalid data URL format"}
        else:
            encoded = image_data
        
        # Decode base64
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        
        # Decode image
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {"error": "Failed to decode image"}
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Inference (optimized with torch.no_grad already in model, now includes severity)
        boxes, labels, scores, severities = model.predict(frame_rgb)
        
        logger.info(f"[HTTP] Detected {len(boxes)} objects")
        
        # Store result with class names and severities
        class_names_list = [CLASS_NAMES.get(label, f"Class {label}") for label in labels]
        latest_frame_result = {
            "boxes": boxes,
            "labels": labels,
            "class_names": class_names_list,
            "scores": scores,
            "severities": severities,
            "image": image_data,
            "timestamp": datetime.now().isoformat()
        }
        
        # Broadcast to WebSocket viewers (Bridge HTTP source -> WebSocket viewers)
        await manager.broadcast(latest_frame_result)
        logger.info(f"[HTTP] Frame broadcasted to {len(manager.active_connections)} viewers")
        
        return {"status": "ok", "detections": len(boxes)}
        
    except Exception as e:
        logger.error(f"[HTTP] Error: {e}")
        return {"error": str(e)}
    finally:
        is_processing = False

@app.get("/api/latest")
async def get_latest_frame():
    """HTTP fallback endpoint for viewer to poll for latest frame"""
    return latest_frame_result

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
