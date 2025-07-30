"""
State Visualizer for Three.js Renderer

Transforms environment state into visualization data for Three.js.
"""

import logging
import numpy as np
import json
from typing import Dict, Any, Optional

logger = logging.getLogger(__name__)

class StateVisualizer:
    """
    Transforms SmartDelivery environment state into Three.js visualization data.
    
    Features:
    - School building visualization
    - Textbook and guide representations
    - Truck delivery animations
    - Real-time state updates
    """
    
    def __init__(self):
        """Initialize the state visualizer."""
        self.school_positions = [
            {'x': -50, 'z': -30, 'name': 'School A'},
            {'x': 50, 'z': -30, 'name': 'School B'},
            {'x': 0, 'z': 50, 'name': 'School C'},
            {'x': -80, 'z': 20, 'name': 'School D'},
            {'x': 80, 'z': 20, 'name': 'School E'}
        ]
        
        self.truck_positions = [
            {'x': -100, 'z': -100, 'name': 'Truck 1'},
            {'x': 100, 'z': -100, 'name': 'Truck 2'},
            {'x': 0, 'z': -150, 'name': 'Truck 3'}
        ]
    
    def _convert_numpy_types(self, obj):
        """Convert NumPy types to Python native types for JSON serialization."""
        try:
            if isinstance(obj, np.integer):
                logger.debug(f"Converting numpy integer: {obj} (type: {type(obj)})")
                return int(obj)
            elif isinstance(obj, np.floating):
                logger.debug(f"Converting numpy float: {obj} (type: {type(obj)})")
                return float(obj)
            elif isinstance(obj, np.ndarray):
                logger.debug(f"Converting numpy array: {obj} (type: {type(obj)})")
                return obj.tolist()
            elif isinstance(obj, dict):
                logger.debug(f"Converting dict with {len(obj)} items")
                return {key: self._convert_numpy_types(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                logger.debug(f"Converting list with {len(obj)} items")
                return [self._convert_numpy_types(item) for item in obj]
            else:
                return obj
        except Exception as e:
            logger.error(f"Error converting numpy types: {e}")
            return obj
    
    def _ensure_json_serializable(self, obj):
        """Ensure object is JSON serializable by converting all problematic types."""
        try:
            # Test JSON serialization
            json.dumps(obj)
            return obj
        except (TypeError, ValueError) as e:
            logger.warning(f"Object not JSON serializable, converting: {e}")
            logger.debug(f"Problematic object type: {type(obj)}")
            if isinstance(obj, dict):
                logger.debug(f"Dict keys: {list(obj.keys())}")
                for key, value in obj.items():
                    if hasattr(value, 'dtype'):  # NumPy type
                        logger.debug(f"Found numpy type in key '{key}': {value} (type: {type(value)})")
            return self._convert_numpy_types(obj)
    
    def transform_state(self, state: Dict[str, Any], action: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Transform environment state into Three.js visualization data for ONE school optimization.
        
        Args:
            state: Environment state dictionary (ONE school)
            action: Optional action that led to this state
            
        Returns:
            Dictionary containing visualization data
        """
        try:
            logger.info(f"Starting state transformation for ONE school...")
            logger.info(f"Input state type: {type(state)}")
            logger.info(f"Input state keys: {list(state.keys()) if isinstance(state, dict) else 'Not a dict'}")
            
            # Validate input state
            if not isinstance(state, dict):
                logger.error(f"State must be a dictionary, got {type(state)}")
                return {'error': 'Invalid state format'}
            
            # Convert all NumPy types to Python native types
            logger.info("Converting NumPy types...")
            state = self._convert_numpy_types(state)
            if action:
                action = self._convert_numpy_types(action)
            
            logger.info("Extracting state variables...")
            # Extract key state variables for ONE school
            num_students = int(state.get('num_students', 0))
            urgency_level = int(state.get('urgency_level', 0))
            textbooks = {
                'kiny': int(state.get('textbooks_kiny', 0)),
                'eng': int(state.get('textbooks_eng', 0)),
                'math': int(state.get('textbooks_math', 0))
            }
            guides = {
                'kiny': int(state.get('guides_kiny', 0)),
                'eng': int(state.get('guides_eng', 0)),
                'math': int(state.get('guides_math', 0))
            }
            quality = {
                'kiny': int(state.get('quality_kiny', 50)),
                'eng': int(state.get('quality_eng', 50)),
                'math': int(state.get('quality_math', 50))
            }
            
            logger.info(f"Extracted values - Students: {num_students}, Textbooks: {textbooks}")
            
            # Determine urgency color
            urgency_colors = {
                0: '#00ff00',  # Green - Low urgency
                1: '#ffff00',  # Yellow - Medium urgency  
                2: '#ff0000'   # Red - High urgency
            }
            urgency_color = urgency_colors.get(urgency_level, '#ff00ff')
            
            logger.info("Creating visualization data for ONE school...")
            
            # Calculate updated resources if action is being executed
            updated_textbooks = textbooks.copy()
            updated_guides = guides.copy()
            
            if action:
                logger.info(f"Processing action: {action}")
                # Apply action to current resources to show immediate effect
                if len(action) >= 6:
                    # Action format: [textbook_kiny, textbook_eng, textbook_math, guide_kiny, guide_eng, guide_math]
                    updated_textbooks['kiny'] += int(action[0])
                    updated_textbooks['eng'] += int(action[1])
                    updated_textbooks['math'] += int(action[2])
                    updated_guides['kiny'] += int(action[3])
                    updated_guides['eng'] += int(action[4])
                    updated_guides['math'] += int(action[5])
                    
                    logger.info(f"Updated textbooks: {updated_textbooks}")
                    logger.info(f"Updated guides: {updated_guides}")
            
            # Create ONE school with detailed state (use updated resources if action is present)
            current_textbooks = updated_textbooks if action else textbooks
            current_guides = updated_guides if action else guides
            
            school_data = {
                'id': 'target_school',
                'name': 'Target School',
                'position': {'x': 0, 'y': 0, 'z': 0},  # Center of scene
                'students': num_students,
                'urgency': urgency_level,
                'urgencyColor': urgency_color,
                'textbooks': current_textbooks,  # Use updated resources
                'guides': current_guides,        # Use updated resources
                'originalTextbooks': textbooks,  # Keep original for reference
                'originalGuides': guides,        # Keep original for reference
                'quality': quality,
                'infrastructure': int(state.get('infrastructure_rating', 50)),
                'locationType': int(state.get('location_type', 0)),
                'grantUsage': int(state.get('grant_usage', 0)),
                'timeSinceLastDelivery': int(state.get('time_since_last_delivery', 0)),
                'deliverySuccessHistory': int(state.get('delivery_success_history', 0)),
                'hasActiveDelivery': bool(action)
            }

            # Create delivery truck with lifecycle animation
            truck_data = {
                'id': 'delivery_truck',
                'name': 'Delivery Truck',
                'position': {'x': -50, 'y': 1, 'z': 0},  # Start position
                'targetPosition': {'x': 0, 'y': 1, 'z': 0},  # School position
                'status': 'delivering' if action else 'idle',
                'cargo': {
                    'textbooks': textbooks,  # Original cargo
                    'guides': guides
                },
                'action': action,
                'deliveryPhase': 'approaching' if action else 'idle'
            }
            
            # Create individual textbooks stacked at the back of the school with headers
            textbooks_data = []
            subject_positions = {
                'kiny': {'x': -15, 'z': -25, 'header': 'Kinyarwanda'},
                'eng': {'x': 0, 'z': -25, 'header': 'English'},
                'math': {'x': 15, 'z': -25, 'header': 'Mathematics'}
            }
            
            # Use updated textbooks for visualization
            for subject, count in current_textbooks.items():
                if count > 0:
                    # Create stacked textbook objects at the back of the school
                    max_visual_books = min(count, 25)  # Show up to 25 visual books
                    subject_pos = subject_positions.get(subject, {'x': 0, 'z': -25, 'header': subject.title()})
                    
                    # Add header for this subject
                    header_data = {
                        'id': f'header_{subject}',
                        'type': 'header',
                        'subject': subject,
                        'text': subject_pos['header'],
                        'position': {
                            'x': subject_pos['x'],
                            'y': 3.5,
                            'z': subject_pos['z']
                        },
                        'color': '#FFFFFF',
                        'fontSize': 16
                    }
                    textbooks_data.append(header_data)
                    
                    for i in range(max_visual_books):
                        # Calculate stacking position - evenly spaced but not too far apart
                        books_per_row = 5
                        books_per_layer = 5
                        layer = i // (books_per_row * books_per_layer)
                        row = (i % (books_per_row * books_per_layer)) // books_per_layer
                        position_in_layer = i % books_per_layer
                        
                        textbook_data = {
                            'id': f'textbook_{subject}_{i}',
                            'subject': subject,
                            'position': {
                                'x': subject_pos['x'] + (position_in_layer - 2) * 1.5,  # Spread in layer
                                'y': 0.1 + layer * 0.4,  # Stack vertically
                                'z': subject_pos['z'] + (row - 2) * 1.5  # Spread in rows
                            },
                            'quality': int(quality.get(subject, 50)),
                            'color': self._get_subject_color(subject),
                            'status': 'stored',
                            'totalCount': count,  # Show actual total
                            'visualCount': max_visual_books  # Show visual count
                        }
                        textbooks_data.append(textbook_data)
            
            # Create individual guides stacked at the front of the school
            guides_data = []
            guide_subject_positions = {
                'kiny': {'x': -12, 'z': 25, 'header': 'Kinyarwanda Guides'},
                'eng': {'x': 0, 'z': 25, 'header': 'English Guides'},
                'math': {'x': 12, 'z': 25, 'header': 'Mathematics Guides'}
            }
            
            # Use updated guides for visualization
            for subject, count in current_guides.items():
                if count > 0:
                    # Create stacked guide objects at the front of the school
                    max_visual_guides = min(count, 20)  # Show up to 20 visual guides
                    subject_pos = guide_subject_positions.get(subject, {'x': 0, 'z': 25, 'header': f'{subject.title()} Guides'})
                    
                    # Add header for this subject
                    header_data = {
                        'id': f'guide_header_{subject}',
                        'type': 'header',
                        'subject': subject,
                        'text': subject_pos['header'],
                        'position': {
                            'x': subject_pos['x'],
                            'y': 3.5,
                            'z': subject_pos['z']
                        },
                        'color': '#FFFFFF',
                        'fontSize': 14
                    }
                    guides_data.append(header_data)
                    
                    for i in range(max_visual_guides):
                        # Calculate stacking position - evenly spaced but not too far apart
                        guides_per_row = 4
                        guides_per_layer = 4
                        layer = i // (guides_per_row * guides_per_layer)
                        row = (i % (guides_per_row * guides_per_layer)) // guides_per_layer
                        position_in_layer = i % guides_per_layer
                        
                        guide_data = {
                            'id': f'guide_{subject}_{i}',
                            'subject': subject,
                            'position': {
                                'x': subject_pos['x'] + (position_in_layer - 1.5) * 1.2,  # Spread in layer
                                'y': 0.075 + layer * 0.3,  # Stack vertically
                                'z': subject_pos['z'] + (row - 1.5) * 1.2  # Spread in rows
                            },
                            'color': self._get_subject_color(subject),
                            'status': 'stored',
                            'totalCount': count,  # Show actual total
                            'visualCount': max_visual_guides  # Show visual count
                        }
                        guides_data.append(guide_data)
            
            # Create delivery animation data
            delivery_animation = {
                'phase': 'idle',
                'truckPosition': truck_data['position'],
                'schoolPosition': school_data['position'],
                'deliveryProgress': 0.0,
                'action': action
            }
            
            # Create visualization data
            visualization_data = {
                'timestamp': float(self._get_timestamp()),
                'updateTime': self._get_formatted_time(),
                'environment': {
                    'numStudents': int(num_students),
                    'urgencyLevel': int(urgency_level),
                    'urgencyColor': urgency_color,
                    'timeSinceLastDelivery': int(state.get('time_since_last_delivery', 0)),
                    'deliverySuccessHistory': int(state.get('delivery_success_history', 0)),
                    'grantUsage': int(state.get('grant_usage', 0)),
                    'infrastructureRating': int(state.get('infrastructure_rating', 50)),
                    'locationType': int(state.get('location_type', 0))
                },
                'school': school_data,  # ONE school
                'textbooks': textbooks_data,
                'guides': guides_data,
                'truck': truck_data,
                'deliveryAnimation': delivery_animation,
                'action': action,
                'hasActiveAction': bool(action)
            }
            
            # Ensure the final data is JSON serializable
            logger.info("Ensuring JSON serializability...")
            visualization_data = self._ensure_json_serializable(visualization_data)
            
            logger.info(f"State transformation completed successfully")
            return visualization_data
            
        except Exception as e:
            logger.error(f"Error transforming state: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return {'error': str(e)}
    
    def _get_subject_color(self, subject: str) -> str:
        """Get color for a subject."""
        colors = {
            'kiny': '#8B4513',  # Brown
            'eng': '#4169E1',   # Blue
            'math': '#32CD32'   # Green
        }
        return colors.get(subject, '#ffffff')
    
    def _get_timestamp(self) -> float:
        """Get current timestamp."""
        import time
        return time.time()
    
    def _get_formatted_time(self) -> str:
        """Get formatted current time."""
        import time
        return time.strftime("%H:%M:%S", time.localtime()) 