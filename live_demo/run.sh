#!/bin/bash
# Quick start script for Linux/Mac

echo "========================================"
echo "Urban Issue Detection - Quick Start"
echo "========================================"
echo ""

# Check if .env exists
if [ ! -f ".env" ]; then
    echo "[INFO] Creating .env file from template..."
    cp env.template .env
    echo ""
    echo "[INFO] Edit .env and add your NGROK_AUTHTOKEN for remote access"
    echo "       Get token at: https://dashboard.ngrok.com/get-started/your-authtoken"
    echo ""
    read -p "Press Enter to continue..."
fi

echo "[INFO] Starting services..."
echo ""

# Start Docker Compose
docker-compose up

# If user pressed Ctrl+C, clean up
echo ""
echo "[INFO] Stopping services..."
docker-compose down
