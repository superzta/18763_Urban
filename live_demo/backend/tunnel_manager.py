"""
Tunnel Manager using ngrok to expose local server to the internet
"""
import os
import asyncio
from typing import Optional, Dict
from pyngrok import ngrok, conf
import logging

logger = logging.getLogger(__name__)

class TunnelManager:
    def __init__(self):
        self.backend_url: Optional[str] = None
        self.frontend_url: Optional[str] = None
        self.backend_tunnel = None
        self.frontend_tunnel = None
        
    def start_tunnels(self, backend_port: int = 8000, frontend_port: int = 3000) -> Dict[str, Optional[str]]:
        """
        Start ngrok tunnel for frontend container (frontend proxies to backend internally)
        Returns dict with backend_url and frontend_url
        """
        try:
            # Set ngrok auth token if provided
            auth_token = os.environ.get("NGROK_AUTHTOKEN")
            if auth_token and auth_token.strip():
                ngrok.set_auth_token(auth_token)
                logger.info("Ngrok auth token set")
            else:
                logger.warning("No NGROK_AUTHTOKEN found. Running in local network only mode.")
                return {"backend_url": None, "frontend_url": None}
            
            # In Docker, frontend runs in a separate container
            # We need to tunnel to 'frontend:3000' not 'localhost:3000'
            # Use the Docker service name 'frontend' to reach the frontend container
            frontend_addr = "frontend:3000"
            
            logger.info(f"Creating ngrok tunnel to {frontend_addr}...")
            self.frontend_tunnel = ngrok.connect(frontend_addr, bind_tls=True)
            self.frontend_url = self.frontend_tunnel.public_url
            
            # Backend is accessible through the same URL via Vite proxy
            self.backend_url = self.frontend_url
            
            logger.info(f"[SUCCESS] Remote access enabled from anywhere!")
            logger.info(f"[INFO] Public URL: {self.frontend_url}")
            logger.info(f"[INFO] Frontend: {self.frontend_url}")
            logger.info(f"[INFO] Backend API: {self.frontend_url}/api/*")
            logger.info(f"[INFO] Share this URL with your phone: {self.frontend_url}")
            
            return {
                "backend_url": self.backend_url,
                "frontend_url": self.frontend_url
            }
            
        except Exception as e:
            logger.error(f"Failed to start ngrok tunnel: {e}")
            logger.info("Falling back to local network only mode")
            return {"backend_url": None, "frontend_url": None}
    
    def get_backend_url(self) -> Optional[str]:
        """Get the backend public URL"""
        return self.backend_url
    
    def get_frontend_url(self) -> Optional[str]:
        """Get the frontend public URL"""
        return self.frontend_url
    
    def stop_tunnels(self):
        """Stop all ngrok tunnels"""
        if self.frontend_tunnel:
            try:
                ngrok.disconnect(self.frontend_tunnel.public_url)
                logger.info("Ngrok tunnel closed")
            except Exception as e:
                logger.error(f"Error closing tunnel: {e}")
            self.frontend_tunnel = None
            self.frontend_url = None
            self.backend_url = None

# Global tunnel manager instance
tunnel_manager = TunnelManager()

