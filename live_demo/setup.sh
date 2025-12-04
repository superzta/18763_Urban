#!/bin/bash

echo "========================================"
echo "Urban Issue Detection - Live Demo Setup"
echo "========================================"
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "[ERROR] Docker is not installed. Please install Docker first."
    echo "        Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! command -v docker-compose &> /dev/null && ! docker compose version &> /dev/null; then
    echo "[ERROR] Docker Compose is not installed. Please install Docker Compose first."
    exit 1
fi

echo "[OK] Docker is installed"
echo ""

# Check for .env file
if [ ! -f ".env" ]; then
    echo "[INFO] Creating .env file from template..."
    cp env.template .env
    echo "[INFO] Please edit .env and add your NGROK_AUTHTOKEN for remote access"
    echo "       Get your token at: https://dashboard.ngrok.com/get-started/your-authtoken"
    echo ""
    read -p "Press Enter to continue with local network only, or Ctrl+C to exit and configure .env first..."
fi

# Check if checkpoints exist
echo "[INFO] Checking for model checkpoints..."
if [ ! -f "../checkpoints/best_model_rcnn_v2.pth" ] && [ ! -f "../checkpoints/fcos_best_model.pth" ]; then
    echo "[WARN] No model checkpoints found in ../checkpoints/"
    echo "       Make sure you have trained models available"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

echo "[OK] Checkpoints found"
echo ""

# Build and start services
echo "[INFO] Building Docker images (this may take 10-15 minutes first time)..."
echo ""

# Use docker compose or docker-compose depending on what's available
if docker compose version &> /dev/null; then
    DOCKER_COMPOSE="docker compose"
else
    DOCKER_COMPOSE="docker-compose"
fi

# Set DOCKER_BUILDKIT for better caching
export DOCKER_BUILDKIT=1
export COMPOSE_DOCKER_CLI_BUILD=1

$DOCKER_COMPOSE build

if [ $? -ne 0 ]; then
    echo "[ERROR] Build failed!"
    exit 1
fi

echo ""
echo "[OK] Build complete!"
echo ""
echo "[INFO] Starting services..."
echo ""

$DOCKER_COMPOSE up -d

if [ $? -ne 0 ]; then
    echo "[ERROR] Failed to start services!"
    exit 1
fi

echo ""
echo "[INFO] Waiting for services to be ready..."
sleep 5

# Check if services are running
if $DOCKER_COMPOSE ps | grep -q "Up"; then
    echo ""
    echo "[SUCCESS] Services are running!"
    echo ""
    echo "Access the application:"
    echo "  - Frontend: http://localhost:3000"
    echo "  - Backend API: http://localhost:8000"
    echo ""
    echo "Remote Access:"
    echo "  Check the backend logs for the ngrok URL (if configured)"
    echo "  Run: $DOCKER_COMPOSE logs backend | grep ngrok"
    echo ""
    echo "View logs:"
    echo "  - All services: $DOCKER_COMPOSE logs -f"
    echo "  - Backend only: $DOCKER_COMPOSE logs -f backend"
    echo "  - Frontend only: $DOCKER_COMPOSE logs -f frontend"
    echo ""
    echo "To stop:"
    echo "  $DOCKER_COMPOSE down"
    echo ""
else
    echo "[ERROR] Services failed to start. Check logs with: $DOCKER_COMPOSE logs"
    exit 1
fi
