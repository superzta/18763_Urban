# PowerShell setup script for Windows

Write-Host "========================================" -ForegroundColor Cyan
Write-Host "Urban Issue Detection - Live Demo Setup" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Check if Docker is installed
try {
    $dockerVersion = docker --version
    Write-Host "[OK] Docker is installed: $dockerVersion" -ForegroundColor Green
} catch {
    Write-Host "[ERROR] Docker is not installed. Please install Docker Desktop first." -ForegroundColor Red
    Write-Host "        Visit: https://docs.docker.com/desktop/install/windows-install/" -ForegroundColor Yellow
    exit 1
}

Write-Host ""

# Check for .env file
if (-Not (Test-Path ".env")) {
    Write-Host "[INFO] Creating .env file from template..." -ForegroundColor Yellow
    Copy-Item "env.template" ".env"
    Write-Host "[INFO] Please edit .env and add your NGROK_AUTHTOKEN for remote access" -ForegroundColor Yellow
    Write-Host "       Get your token at: https://dashboard.ngrok.com/get-started/your-authtoken" -ForegroundColor Cyan
    Write-Host ""
    $continue = Read-Host "Press Enter to continue with local network only, or Ctrl+C to exit and configure .env first"
}

# Check if checkpoints exist
Write-Host "[INFO] Checking for model checkpoints..." -ForegroundColor Cyan
if (-Not (Test-Path "../checkpoints/best_model_rcnn_v2.pth") -and -Not (Test-Path "../checkpoints/fcos_best_model.pth")) {
    Write-Host "[WARN] No model checkpoints found in ../checkpoints/" -ForegroundColor Yellow
    Write-Host "       Make sure you have trained models available" -ForegroundColor Yellow
    $continue = Read-Host "Continue anyway? (y/n)"
    if ($continue -ne "y") {
        exit 1
    }
}

Write-Host "[OK] Checkpoints found" -ForegroundColor Green
Write-Host ""

# Build and start services
Write-Host "[INFO] Building Docker images (this may take 10-15 minutes first time)..." -ForegroundColor Cyan
Write-Host ""

# Set environment variables for BuildKit
$env:DOCKER_BUILDKIT = "1"
$env:COMPOSE_DOCKER_CLI_BUILD = "1"

docker-compose build

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Build failed!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[OK] Build complete!" -ForegroundColor Green
Write-Host ""
Write-Host "[INFO] Starting services..." -ForegroundColor Cyan
Write-Host ""

docker-compose up -d

if ($LASTEXITCODE -ne 0) {
    Write-Host "[ERROR] Failed to start services!" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "[INFO] Waiting for services to be ready..." -ForegroundColor Yellow
Start-Sleep -Seconds 5

# Check if services are running
$running = docker-compose ps | Select-String "Up"
if ($running) {
    Write-Host ""
    Write-Host "[SUCCESS] Services are running!" -ForegroundColor Green
    Write-Host ""
    Write-Host "Access the application:" -ForegroundColor Cyan
    Write-Host "  - Frontend: http://localhost:3000" -ForegroundColor White
    Write-Host "  - Backend API: http://localhost:8000" -ForegroundColor White
    Write-Host ""
    Write-Host "Remote Access:" -ForegroundColor Cyan
    Write-Host "  Check the backend logs for the ngrok URL (if configured)" -ForegroundColor White
    Write-Host "  Run: docker-compose logs backend | Select-String ngrok" -ForegroundColor Yellow
    Write-Host ""
    Write-Host "View logs:" -ForegroundColor Cyan
    Write-Host "  - All services: docker-compose logs -f" -ForegroundColor White
    Write-Host "  - Backend only: docker-compose logs -f backend" -ForegroundColor White
    Write-Host "  - Frontend only: docker-compose logs -f frontend" -ForegroundColor White
    Write-Host ""
    Write-Host "To stop:" -ForegroundColor Cyan
    Write-Host "  docker-compose down" -ForegroundColor White
    Write-Host ""
} else {
    Write-Host "[ERROR] Services failed to start. Check logs with: docker-compose logs" -ForegroundColor Red
    exit 1
}
