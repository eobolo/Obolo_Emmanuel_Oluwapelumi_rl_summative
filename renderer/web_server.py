"""
Web Server for Three.js Renderer

Handles real-time communication between the Python environment
and the Three.js frontend via WebSocket.
"""

import logging
import threading
import time
import json
from flask import Flask, render_template
from flask_socketio import SocketIO, emit
import json

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class WebServer:
    """Flask web server with Socket.IO for real-time communication."""
    
    def __init__(self, host='127.0.0.1', port=5000):
        """Initialize the web server."""
        self.host = host
        self.port = port
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = 'smartdelivery_secret'
        
        # Initialize SocketIO with working configuration
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*"
        )
        
        self.server_thread = None
        self.is_running = False
        
        # Register routes
        self._register_routes()
        
        logger.info(f"WebServer initialized on {self.host}:{self.port}")
    
    def _register_routes(self):
        """Register Flask routes."""
        
        @self.app.route('/')
        def index():
            """Serve the main HTML page."""
            return render_template('index.html')  # Use the proper 3D visualization template
        
        @self.socketio.on('connect')
        def handle_connect():
            """Handle client connection."""
            logger.info("Client connected to WebSocket")
            # Send a connected event to confirm connection
            emit('connected', {'status': 'connected'})
            logger.info("Sent connected event to client")
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            """Handle client disconnection."""
            logger.info("Client disconnected from WebSocket")
        
        @self.socketio.on('request_state')
        def handle_request_state():
            """Handle state requests from the client."""
            logger.info("Client requested state")
            # Send the latest state if available
            if hasattr(self, 'latest_state') and self.latest_state:
                logger.info("Sending latest state to client")
                emit('state_update', self.latest_state)
            else:
                logger.warning("No state available to send")
                emit('state_update', {'error': 'No state available'})
    
    def start(self):
        """Start the web server in a separate thread."""
        if self.is_running:
            logger.warning("Web server is already running")
            return
        
        def run_server():
            try:
                logger.info(f"Web server starting at http://{self.host}:{self.port}")
                self.socketio.run(
                    self.app,
                    host=self.host,
                    port=self.port,
                    debug=False
                )
            except Exception as e:
                logger.error(f"Web server error: {e}")
                self.is_running = False
        
        self.server_thread = threading.Thread(target=run_server, daemon=True)
        self.server_thread.start()
        
        # Wait a moment for server to start
        time.sleep(3)
        self.is_running = True
        logger.info(f"Web server started at http://{self.host}:{self.port}")
    
    def stop(self):
        """Stop the web server."""
        if not self.is_running:
            return
        
        try:
            self.socketio.stop()
            self.is_running = False
            logger.info("Web server stopped")
        except Exception as e:
            logger.error(f"Error stopping web server: {e}")
    
    def _ensure_json_serializable(self, data):
        """Ensure data is JSON serializable."""
        try:
            # Test JSON serialization
            json.dumps(data)
            return data
        except (TypeError, ValueError) as e:
            logger.error(f"Data not JSON serializable: {e}")
            # Try to convert problematic types
            if isinstance(data, dict):
                return {k: self._ensure_json_serializable(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [self._ensure_json_serializable(item) for item in data]
            else:
                return str(data)
    
    def emit_state(self, state_data):
        """Emit state data to connected clients."""
        if not self.is_running:
            logger.warning("Web server not running, cannot emit state")
            return
        
        try:
            logger.info(f"Attempting to emit state data...")
            logger.info(f"State data type: {type(state_data)}")
            logger.info(f"State data keys: {list(state_data.keys()) if isinstance(state_data, dict) else 'Not a dict'}")
            
            # Ensure data is JSON serializable
            state_data = self._ensure_json_serializable(state_data)
            self.latest_state = state_data  # Store latest state for request_state
            
            # Test JSON serialization before emitting
            try:
                json_str = json.dumps(state_data)
                logger.info(f"JSON serialization successful, length: {len(json_str)}")
            except Exception as e:
                logger.error(f"JSON serialization failed: {e}")
                return
            
            self.socketio.emit('state_update', state_data)
            logger.info(f"State update emitted successfully to clients")
            
        except Exception as e:
            logger.error(f"Error emitting state: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
    
    def get_url(self):
        """Get the server URL."""
        return f"http://{self.host}:{self.port}"