# Urban Issue Detection - Live Demo

Real-time object detection for road-related issues using R-CNN or FCOS models with remote phone camera access from anywhere in the world.

## Features

- Video Upload & Processing: Upload videos and get detected objects with bounding boxes
- Live Phone Camera: Connect your phone camera from anywhere (not just local network)
- Remote Access: Uses ngrok tunneling for global accessibility
- Real-time Detection: See bounding boxes, class names, and confidence scores live
- Optimized Docker: Fast rebuilds with smart layer caching (30 seconds vs 20 minutes)
- Hot Reload: Changes to code update instantly without full rebuild

## Prerequisites

### Required
- Docker Desktop (Windows: https://docs.docker.com/desktop/install/windows-install/)
- Model checkpoints in `../checkpoints/` directory
- 8GB+ RAM recommended

### Optional (for remote access from anywhere)
- Ngrok Account (free): https://dashboard.ngrok.com/signup
  - Without ngrok: Works only on local network (same Wi-Fi)
  - With ngrok: Works from anywhere in the world

---

## QUICK START - 3 STEPS

### Step 1: Get Ngrok Token (Optional, 2 minutes)

If you want remote access from anywhere:

1. Visit: https://dashboard.ngrok.com/signup (free, no credit card)
2. After login: https://dashboard.ngrok.com/get-started/your-authtoken
3. Copy your token (format: `2abc...xyz123`)

### Step 2: Setup (1 minute)

Open PowerShell in the `live_demo` folder:

```powershell
# Create environment file
copy env.template .env

# Edit .env in notepad and paste your ngrok token (or skip for local-only)
notepad .env
```

### Step 3: Run (5-10 minutes first time, 30 seconds after)

```powershell
# Run setup script
.\setup.ps1
```

Or manually:

```powershell
# First time (builds everything)
docker-compose up --build

# Subsequent runs (much faster, uses cache)
docker-compose up

# Run in background
docker-compose up -d

# Stop services
docker-compose down
```

### Access the Application

After starting:
- Frontend: http://localhost:3000
- Backend API: http://localhost:8000
- API Docs: http://localhost:8000/docs

If ngrok is configured, check backend logs for public URL:
```powershell
docker-compose logs backend | Select-String "tunnel"

# You should see:
# [SUCCESS] Remote access enabled from anywhere!
# [INFO] Public URL: https://abc123.ngrok.io
# [INFO] Frontend: https://abc123.ngrok.io
# [INFO] Backend API: https://abc123.ngrok.io/api/*
```

---

## USAGE GUIDE

### 1. Video Inference

1. Open http://localhost:3000 in browser
2. Click "Video Inference" tab
3. Select model checkpoint (R-CNN or FCOS)
4. Upload a video file
5. Click "Start Inference"
6. Wait for processing
7. Download processed video with bounding boxes

### 2. Live Camera - Local Network Only

Requirements: Phone and laptop on same Wi-Fi network

1. On laptop: Open http://localhost:3000
2. Go to "Live Camera" tab
3. You'll see: "Local Network Only" badge (yellow)
4. Scan QR code with phone camera app
5. Browser opens automatically
6. Allow camera permission
7. See real-time detections on laptop

### 3. Live Camera - Remote Access (Recommended)

Requirements: Ngrok token configured in .env

1. On laptop: Open http://localhost:3000
2. Go to "Live Camera" tab
3. You'll see: "Remote Access Enabled" badge (green)
4. Scan QR code with phone from ANYWHERE (different city, cellular data, etc.)
5. Browser opens automatically
6. Allow camera permission
7. See real-time detections on laptop

The QR code contains the public ngrok URL that works from anywhere in the world.

---

## TROUBLESHOOTING

### Issue: Docker rebuild takes 20 minutes every time

Solution: Don't use `--build` flag unless needed

```powershell
# WRONG (rebuilds everything)
docker-compose up --build

# CORRECT (uses cache, 30 seconds)
docker-compose up

# Only rebuild when Dockerfile changes
docker-compose build
docker-compose up
```

### Issue: Frontend build fails with "npm ci" error

Solution: Already fixed! The Dockerfile now uses `npm install` instead.

If still failing:
```powershell
# Delete node_modules and try again
docker-compose down -v
docker-compose build --no-cache frontend
docker-compose up
```

### Issue: QR code shows localhost:3000, phone can't connect

Solution: Enable remote access with ngrok

1. Get token: https://dashboard.ngrok.com/get-started/your-authtoken
2. Add to .env:
   ```
   NGROK_AUTHTOKEN=your_actual_token_here
   ```
3. Restart services:
   ```powershell
   docker-compose down
   docker-compose up
   ```

### Issue: Phone still can't connect even with ngrok

Solution: 
1. Check backend logs for ngrok tunnel:
   ```powershell
   docker-compose logs backend | Select-String "tunnel"
   ```
2. Verify you see the public URL:
   - [INFO] Public URL: https://abc123.ngrok.io
3. Check token is correct in .env (no extra spaces, no quotes)
4. Try accessing the ngrok URL in laptop browser first
5. Make sure services restarted after adding token:
   ```powershell
   docker-compose down
   docker-compose up
   ```

### Issue: "Checkpoint not found" error

Solution: Ensure model checkpoints exist

```powershell
# Check if checkpoints exist
dir ..\checkpoints\

# Should show:
# best_model_rcnn_v2.pth
# fcos_best_model.pth

# Copy your trained models if missing
copy path\to\your\model ..\checkpoints\
```

### Issue: Port already in use (3000 or 8000)

Solution:

```powershell
# Find process using port
netstat -ano | findstr :3000
netstat -ano | findstr :8000

# Kill the process (replace PID with actual process ID)
taskkill /PID <PID> /F

# Or stop all docker services
docker-compose down
```

### Issue: Out of memory / Docker crash

Solution:

1. Increase Docker Desktop memory:
   - Open Docker Desktop
   - Settings > Resources > Memory
   - Increase to 6-8 GB
   - Apply & Restart

2. Or limit container memory in docker-compose.yml:
   ```yaml
   services:
     backend:
       mem_limit: 4g
   ```

---

## SYSTEM ARCHITECTURE

```
┌─────────────────────────────────────────────────┐
│                   ANYWHERE                      │
│  ┌─────────────────────────────────────────┐   │
│  │        Phone (Mobile Browser)           │   │
│  │      WebSocket + Video Stream           │   │
│  └──────────────┬──────────────────────────┘   │
└─────────────────┼──────────────────────────────┘
                  │
                  ↓ HTTPS/WSS
         ┌────────────────────┐
         │  Ngrok Tunnel      │
         │  (Public Internet) │
         └────────┬───────────┘
                  │
                  ↓
    ┌─────────────────────────────────────┐
    │        Your Windows Laptop          │
    │  ┌────────────────────────────┐     │
    │  │  Frontend (React + Vite)   │     │
    │  │  Port: 3000                │     │
    │  └────────────┬───────────────┘     │
    │               │                     │
    │  ┌────────────▼───────────────┐     │
    │  │  Backend (FastAPI)         │     │
    │  │  - WebSocket Server        │     │
    │  │  - ML Model Inference      │     │
    │  │  - Ngrok Integration       │     │
    │  │  Port: 8000                │     │
    │  └────────────┬───────────────┘     │
    │               │                     │
    │  ┌────────────▼───────────────┐     │
    │  │  Model Checkpoints         │     │
    │  │  - R-CNN                   │     │
    │  │  - FCOS                    │     │
    │  └────────────────────────────┘     │
    └─────────────────────────────────────┘
```

---

## DOCKER OPTIMIZATION EXPLAINED

### Why is it fast now?

The Dockerfile uses smart layer caching:

1. **Dependencies first** (changes rarely):
   - requirements.txt installation
   - npm packages installation

2. **Model files next** (large, stable):
   - Checkpoint files (500MB+)
   - Model definitions

3. **Code last** (changes frequently):
   - Backend Python code
   - Frontend React code

Result: When you change code, Docker only rebuilds the last layer (30 seconds instead of 20 minutes).

### Build Times

| Scenario | Time | Notes |
|----------|------|-------|
| First build | 10-15 min | Downloads everything |
| Code change + restart | 5-10 sec | Hot reload, no rebuild |
| Code change + rebuild | 30-60 sec | Only affected layers |
| New dependency | 2-5 min | Re-installs packages |
| Clean rebuild | 10-15 min | Use `--no-cache` |

---

## COMMON COMMANDS (Windows PowerShell)

### Starting & Stopping

```powershell
# Start services (first time)
docker-compose up --build

# Start services (subsequent times)
docker-compose up

# Start in background
docker-compose up -d

# Stop services
docker-compose down

# Stop and remove volumes
docker-compose down -v
```

### Viewing Logs

```powershell
# All services
docker-compose logs -f

# Backend only
docker-compose logs -f backend

# Frontend only
docker-compose logs -f frontend

# Find ngrok URL
docker-compose logs backend | Select-String "ngrok"
docker-compose logs backend | Select-String "remote"

# Save logs to file
docker-compose logs > logs.txt
```

### Rebuilding

```powershell
# Rebuild specific service
docker-compose build backend
docker-compose build frontend

# Rebuild without cache (clean build)
docker-compose build --no-cache

# Rebuild and start
docker-compose up --build
```

### Checking Status

```powershell
# Check if running
docker-compose ps

# Check resource usage
docker stats

# Check Docker version
docker --version
docker-compose --version
```

### Cleaning Up

```powershell
# Remove old images
docker image prune -a

# Remove unused volumes
docker volume prune

# Complete cleanup (WARNING: removes everything)
docker system prune -a --volumes
```

---

## REMOTE ACCESS DETAILS

### How Ngrok Works

1. Your backend runs on localhost:8000, frontend on localhost:3000
2. Ngrok creates ONE secure tunnel to the frontend (port 3000)
3. Provides one public HTTPS URL: https://abc123.ngrok.io
4. Frontend has a built-in proxy that forwards /api/* requests to backend
5. The QR code shows this URL - phone accesses everything through it
6. Works from anywhere in the world

### Ngrok Free Tier

- Cost: $0
- Bandwidth: 1 GB/month (sufficient for demos)
- Speed: Same as paid
- URL: Changes every restart (1 random URL created)
- Connections: 40/minute limit
- Tunnels: 1 tunnel to frontend (frontend proxies to backend internally)

### Ngrok Paid Tier ($8-10/month)

- Static domain (same URL always)
- More bandwidth
- Higher rate limits
- Custom domains
- Better for permanent deployments

### Getting Public URL

After starting with ngrok token configured:

```powershell
# Check backend logs
docker-compose logs backend | Select-String "tunnel"

# Look for lines:
# [SUCCESS] Remote access enabled from anywhere!
# [INFO] Public URL: https://abc123.ngrok.io
# [INFO] Frontend: https://abc123.ngrok.io
# [INFO] Backend API: https://abc123.ngrok.io/api/*
```

This URL is what appears in the QR code for phone scanning.
Both frontend and API are accessible through the same URL (proxy handles routing).

---

## PERFORMANCE EXPECTATIONS

### Inference Times

| Model | Video (1 min) | Single Frame | Hardware |
|-------|---------------|--------------|----------|
| R-CNN | 2-5 minutes | 100-200ms | CPU |
| FCOS  | 1-3 minutes | 80-150ms | CPU |
| R-CNN | 30-60 sec | 20-50ms | GPU |
| FCOS  | 20-40 sec | 15-30ms | GPU |

### Network Requirements

| Feature | Bandwidth | Latency |
|---------|-----------|---------|
| Video Upload | ~10 Mbps | N/A |
| Live Camera (Local) | ~2-5 Mbps | <50ms |
| Live Camera (Remote) | ~2-5 Mbps | <200ms |

### Frame Rate

- Local network: ~10 FPS (100ms interval)
- Remote access: ~5-10 FPS (depends on connection)

---

## CONFIGURATION FILES

### .env (Environment Variables)

```
NGROK_AUTHTOKEN=your_token_here
VITE_API_URL=http://localhost:8000
```

### docker-compose.yml

Controls how services run:
- Port mappings
- Volume mounts (for hot reload)
- Environment variables
- Resource limits

### Dockerfile.backend

Python backend container:
- Base: python:3.9-slim
- Installs: PyTorch, FastAPI, OpenCV, ngrok
- Exposes: Port 8000

### Dockerfile.frontend

React frontend container:
- Base: node:18-alpine
- Installs: React, Vite, Tailwind
- Exposes: Port 3000

---

## TECHNICAL DETAILS

### Backend (FastAPI)

- Framework: FastAPI (async Python)
- WebSocket: For real-time phone camera stream
- ML: PyTorch models (R-CNN, FCOS)
- Tunnel: pyngrok for remote access
- CORS: Enabled for all origins (demo only)

### Frontend (React + Vite)

- Framework: React 18
- Build: Vite (fast HMR)
- Styling: Tailwind CSS
- WebSocket: Native WebSocket API
- QR Code: qrcode.react library

### Models

Detects road-related issues (classes 0, 1, 3):
- R-CNN: Faster R-CNN with ResNet backbone
- FCOS: Fully Convolutional One-Stage detector
- RetinaNet: Focal loss based detector

Checkpoints: Located in `../checkpoints/`

---

## SECURITY NOTES

### For Demo/Development

Current setup is fine:
- CORS allows all origins
- No authentication
- Public ngrok URL (if enabled)

### For Production

Add these security measures:

1. **Authentication**:
   ```python
   from fastapi.security import HTTPBasic
   security = HTTPBasic()
   ```

2. **Rate Limiting**:
   ```python
   from slowapi import Limiter
   limiter = Limiter(key_func=get_remote_address)
   ```

3. **Restrict CORS**:
   ```python
   allow_origins=["https://your-domain.com"]
   ```

4. **Use HTTPS**:
   - Ngrok provides HTTPS automatically
   - For production: Use nginx with SSL certificate

5. **Environment Variables**:
   - Never commit .env file
   - Use secrets management in production

---

## COST BREAKDOWN

### Free Setup (Recommended for Demos)

- Docker: $0 (uses your computer)
- Ngrok: $0 (free tier)
- Total: $0/month

Limitations:
- Ngrok URL changes on restart
- 1 GB bandwidth/month
- Runs only when laptop is on

### Paid Ngrok ($8-10/month)

Benefits:
- Static URL (doesn't change)
- More bandwidth
- Better for frequent demos

### Production Deployment ($12-50/month)

Options:
- VPS (DigitalOcean, AWS EC2): $12-40/month
- Platform-as-a-Service (Railway, Render): $5-25/month
- Benefits: 24/7 uptime, more resources

---

## DEVELOPMENT TIPS

### Hot Reload

Code changes appear instantly without rebuild:

```powershell
# Backend: Edit files in live_demo/backend/
# Changes auto-reload with uvicorn --reload

# Frontend: Edit files in live_demo/frontend/src/
# Changes auto-reload with Vite HMR
```

### Debugging

```powershell
# Access backend container shell
docker-compose exec backend bash

# Access frontend container shell
docker-compose exec frontend sh

# Check Python packages
docker-compose exec backend pip list

# Check Node packages
docker-compose exec frontend npm list
```

### Model Switching

In the frontend UI:
1. Use model dropdown in header
2. Select: Faster R-CNN, FCOS, or RetinaNet
3. Changes apply immediately to next inference

Or configure in backend:

```python
# live_demo/backend/main.py
current_model_config = {
    "type": "rcnn",  # or "fcos" or "retinanet"
    "checkpoint": "checkpoints/best_model_rcnn_v2.pth",
    "classes": [0, 1, 3]
}
```

---

## FILE STRUCTURE

```
live_demo/
├── backend/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with ngrok
│   ├── model_wrapper.py     # Model loading & inference
│   ├── video_processor.py   # Video processing
│   └── tunnel_manager.py    # Ngrok tunnel manager
├── frontend/
│   ├── src/
│   │   ├── App.jsx          # Main app with dynamic API URL
│   │   ├── main.jsx
│   │   ├── index.css
│   │   └── components/
│   │       ├── VideoInference.jsx
│   │       └── LiveCamera.jsx
│   ├── index.html
│   ├── package.json
│   ├── vite.config.js
│   ├── tailwind.config.js
│   └── postcss.config.js
├── Dockerfile.backend       # Optimized backend container
├── Dockerfile.frontend      # Optimized frontend container
├── docker-compose.yml       # Service orchestration
├── .dockerignore           # Exclude unnecessary files
├── .gitignore             # Git ignore patterns
├── env.template           # Environment template
├── setup.ps1              # Windows setup script
├── setup.sh               # Linux/Mac setup script
├── run.bat                # Windows quick run
├── run.sh                 # Linux/Mac quick run
└── README.md              # This file
```

---

## TESTING CHECKLIST

Before demo:

- [ ] Docker Desktop running
- [ ] Checkpoints in ../checkpoints/ directory
- [ ] .env file created (with ngrok token if needed)
- [ ] Services start successfully
- [ ] Frontend loads at localhost:3000
- [ ] Backend responds at localhost:8000
- [ ] Video inference produces output
- [ ] Live camera connects
- [ ] Phone camera streams
- [ ] Detections appear in real-time
- [ ] Ngrok tunnel works (if configured)

---

## SUPPORT

If you encounter issues:

1. Check logs: `docker-compose logs -f`
2. Verify .env: `type .env`
3. Test connection: Visit http://localhost:8000
4. Review troubleshooting section above
5. Check Docker Desktop is running
6. Ensure ports 3000 and 8000 are free

Common solutions:
- Restart Docker Desktop
- Run `docker-compose down` then `docker-compose up`
- Clear Docker cache: `docker system prune -a`
- Rebuild without cache: `docker-compose build --no-cache`

---

## WHAT WAS FIXED

### 1. Docker Build Speed

- Before: 20 minutes every time
- After: 30 seconds (40x faster)
- How: Smart layer caching, proper layer ordering

### 2. Remote Access

- Before: Only same Wi-Fi network, localhost URLs
- After: Works from anywhere with ngrok
- How: Integrated ngrok tunneling, dynamic URL detection

### 3. Frontend Build

- Before: npm ci fails without package-lock.json
- After: npm install works reliably
- How: Changed to npm install instead of npm ci

### 4. Documentation

- Before: Multiple scattered docs with emojis
- After: Single comprehensive README
- How: Consolidated all guides, removed emojis

---

## LICENSE

See main project LICENSE file.

---

## Success Checklist

You know everything is working when:

1. Services start in under 1 minute (after first build)
2. Frontend accessible at localhost:3000
3. Backend accessible at localhost:8000
4. Video upload produces annotated video
5. QR code scan opens phone camera
6. Phone stream appears on laptop
7. Bounding boxes show in real-time
8. Works from anywhere (with ngrok)

Congratulations! Your urban issue detection system is ready.
