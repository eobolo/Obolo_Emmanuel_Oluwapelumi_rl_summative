"""
Three.js Renderer for SmartDelivery Environment

Provides a web-based 3D visualization using Three.js and Flask.
"""

import logging
import threading
import time
from typing import Dict, Any, Optional

from .web_server import WebServer
from .state_visualizer import StateVisualizer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ThreeJSRenderer:
    """
    Three.js-based renderer for the SmartDelivery environment.
    
    Features:
    - Real-time 3D visualization
    - WebSocket communication
    - Interactive controls
    - Performance monitoring
    """
    
    def __init__(self, host='127.0.0.1', port=5000, auto_start=True):
        """
        Initialize the Three.js renderer.
        
        Args:
            host: Web server host address
            port: Web server port
            auto_start: Whether to start the server automatically
        """
        self.host = host
        self.port = port
        self.auto_start = auto_start
        
        # Initialize components
        self.web_server = WebServer(host=host, port=port)
        self.state_visualizer = StateVisualizer()
        
        # State tracking
        self.current_state = {}
        self.last_action = None
        self.is_running = False
        
        logger.info(f"ThreeJSRenderer initialized on {host}:{port}")
        
        if auto_start:
            self.start()
    
    def start(self):
        """Start the renderer."""
        if self.is_running:
            logger.warning("Renderer is already running")
            return
        
        try:
            self.web_server.start()
            self.is_running = True
            logger.info("ThreeJSRenderer started successfully")
        except Exception as e:
            logger.error(f"Failed to start renderer: {e}")
            raise
    
    def stop(self):
        """Stop the renderer."""
        if not self.is_running:
            return
        
        try:
            self.web_server.stop()
            self.is_running = False
            logger.info("ThreeJSRenderer stopped")
        except Exception as e:
            logger.error(f"Error stopping renderer: {e}")
    
    def update_state(self, state: Dict[str, Any], action: Optional[Dict[str, Any]] = None):
        """
        Update the visualization with new state data.
        
        Args:
            state: Environment state dictionary
            action: Optional action that led to this state
        """
        if not self.is_running:
            logger.warning("Renderer not running, cannot update state")
            return
        
        try:
            # Store current state
            self.current_state = state.copy()
            if action:
                self.last_action = action
            
            # Debug: Log the input state
            logger.info(f"Updating state with keys: {list(state.keys())}")
            logger.info(f"State sample values: {dict(list(state.items())[:3])}")
            
            # Transform state for visualization
            visualization_data = self.state_visualizer.transform_state(state, action)
            
            # Debug: Log the transformed data
            logger.info(f"Transformed data keys: {list(visualization_data.keys()) if isinstance(visualization_data, dict) else 'Not a dict'}")
            
            # Send to web server
            self.web_server.emit_state(visualization_data)
            
            logger.info(f"State update completed successfully")
            
        except Exception as e:
            logger.error(f"Error updating state: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def get_server_url(self) -> str:
        """Get the web server URL."""
        return self.web_server.get_url()
    
    def is_healthy(self) -> bool:
        """Check if the renderer is healthy."""
        return self.is_running and self.web_server.is_running
    
    def __enter__(self):
        """Context manager entry."""
        if not self.is_running and self.auto_start:
            self.start()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop() 