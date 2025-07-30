import numpy as np
import gymnasium as gym
from gymnasium import spaces
import math
import pygame
import os


from step.state_transitions import step
from state.num_of_students_state import numStudentState
from state.textbook_available_state import sample_textbooks_available, verify_textbooks_available
from state.teacher_guide_availability_state import sample_teacher_guide_availability, verify_teacher_guide_availability
from state.textbook_quality_score import sample_textbook_quality_score, verify_textbook_quality_score
from state.grant_usage_state import sample_grant_usage, verify_grant_usage
from state.time_since_last_delivery_state import sample_time_since_last_delivery, verify_time_since_last_delivery
from state.delivery_success_history_state import sample_delivery_success_history, verify_delivery_success_history
from state.urgency_level_state import sample_urgency_level, verify_urgency_level
from state.infrastructure_rating_state import sample_infrastructure_rating, verify_infrastructure_rating
from state.location_type_state import sample_location_type, verify_location_type

# Utility functions for safe conversion
URGENCY_LEVELS = ['Low', 'Medium', 'High']
DELIVERY_SUCCESS_LEVELS = ['Low', 'Medium', 'High']
def urgency_to_str(val):
    if isinstance(val, int):
        return URGENCY_LEVELS[val]
    return val

def urgency_to_int(val):
    if isinstance(val, str):
        return URGENCY_LEVELS.index(val)
    return val

def delivery_success_to_str(val):
    if isinstance(val, int):
        return DELIVERY_SUCCESS_LEVELS[val]
    return val

def delivery_success_to_int(val):
    if isinstance(val, str):
        return DELIVERY_SUCCESS_LEVELS.index(val)
    return val

class P3Environment(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, max_steps=100, render_mode=None):
        super(P3Environment, self).__init__()
        self.max_steps = max_steps
        self.current_step = 0
        self.render_mode = render_mode
        self.num_teachers = np.clip(np.round(np.random.normal(2, 0.5)), 1, 3).astype(int)
        self.state = self._initialize_state()
        self.screen = None
        self.clock = None
        self.truck_angle = 0
        self.truck_frame = 0
        self.blink_frame = 0
        self.paused = False
        self.sprites = {}

        # Define observation space
        self.observation_space = spaces.Dict({
            'num_students': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'textbooks_kiny': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'textbooks_eng': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'textbooks_math': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'guides_kiny': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'guides_eng': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'guides_math': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'quality_kiny': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            'quality_eng': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            'quality_math': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            'grant_usage': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            'time_since_last_delivery': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int32),
            'delivery_success_history': spaces.Discrete(3),  # 0: Low, 1: Medium, 2: High
            'urgency_level': spaces.Discrete(3),  # 0: Low, 1: Medium, 2: High
            'infrastructure_rating': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            'location_type': spaces.Discrete(2)  # 0: Rural, 1: Urban
        })

        # Define discrete action space (5 levels per dimension: 0, 50, 100, 150, 200 for textbooks; 0, 1, 2, 3 for guides)
        n_textbook_levels = 5  # 0, 50, 100, 150, 200
        n_guide_levels = 4    # 0, 1, 2, 3
        self.action_space = spaces.Discrete(n_textbook_levels ** 3 * n_guide_levels ** 3)  # 5^3 * 4^3 = 125 * 64 = 8000 actions

        if self.render_mode == "human":
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            self.clock = pygame.time.Clock()
            sprite_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sprites')
            self.sprites['student'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'student.png')).convert_alpha(), (32, 32))
            self.sprites['textbook_kiny'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'textbook_kiny.png')).convert_alpha(), (16, 16))
            self.sprites['textbook_eng'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'textbook_eng.png')).convert_alpha(), (16, 16))
            self.sprites['textbook_math'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'textbook_math.png')).convert_alpha(), (16, 16))
            self.sprites['teacher'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'teacher.png')).convert_alpha(), (80, 80))
            self.sprites['guide_kiny'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'guide_kiny.png')).convert_alpha(), (16, 16))
            self.sprites['guide_eng'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'guide_eng.png')).convert_alpha(), (16, 16))
            self.sprites['guide_math'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'guide_math.png')).convert_alpha(), (16, 16))
            self.sprites['truck_frame1'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'truck_frame1.png')).convert_alpha(), (48, 32))
            self.sprites['truck_frame2'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'truck_frame2.png')).convert_alpha(), (48, 32))
            self.sprites['truck_frame3'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'truck_frame3.png')).convert_alpha(), (48, 32))
            try:
                self.sprites['background'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'background.png')).convert(), (800, 600))
            except pygame.error:
                print("Warning: background.png not found or invalid. Using gradient background.")
                self.sprites['background'] = pygame.Surface((800, 600))
                for y in range(600):
                    r = int(200 + 55 * math.sin(y * 0.02))
                    r = max(0, min(255, r))  # Clamp r to valid range
                    self.sprites['background'].fill((r, 150, 200), (0, y, 800, 1))

    def is_paused(self):
        """Return the current pause state."""
        return self.paused

    def sample(self):
        # Sample a discrete action index and convert to action tuple
        action_idx = self.action_space.sample()
        n_textbook_levels = 5
        n_guide_levels = 4
        D_kiny = (action_idx // (n_textbook_levels ** 2 * n_guide_levels ** 3)) % n_textbook_levels * 50  # 0, 50, 100, 150, 200
        D_eng = (action_idx // (n_textbook_levels * n_guide_levels ** 3)) % n_textbook_levels * 50
        D_math = (action_idx // n_guide_levels ** 3) % n_textbook_levels * 50
        DG_kiny = (action_idx // (n_guide_levels ** 2)) % n_guide_levels  # 0, 1, 2, 3
        DG_eng = (action_idx // n_guide_levels) % n_guide_levels
        DG_math = action_idx % n_guide_levels
        return (D_kiny, D_eng, D_math, DG_kiny, DG_eng, DG_math)

    @staticmethod
    def decode_action(action_idx):
        n_textbook_levels = 5
        n_guide_levels = 4
        D_kiny = (action_idx // (n_textbook_levels ** 2 * n_guide_levels ** 3)) % n_textbook_levels * 50
        D_eng = (action_idx // (n_textbook_levels * n_guide_levels ** 3)) % n_textbook_levels * 50
        D_math = (action_idx // n_guide_levels ** 3) % n_textbook_levels * 50
        DG_kiny = (action_idx // (n_guide_levels ** 2)) % n_guide_levels
        DG_eng = (action_idx // n_guide_levels) % n_guide_levels
        DG_math = action_idx % n_guide_levels
        return (D_kiny, D_eng, D_math, DG_kiny, DG_eng, DG_math)

    def _initialize_state(self):
        num_students_instance = numStudentState()
        location_type = sample_location_type()  # Returns string
        num_students = int(num_students_instance.sample_num_students())
        t_kiny, t_eng, t_math = sample_textbooks_available(num_students, location_type)
        g_kiny, g_eng, g_math = sample_teacher_guide_availability(location_type, self.num_teachers)
        q_kiny, q_eng, q_math = sample_textbook_quality_score(location_type)
        grant_usage = int(sample_grant_usage(location_type))
        infrastructure_rating = int(sample_infrastructure_rating(location_type))
        time_since_last_delivery = int(sample_time_since_last_delivery())
        delivery_success_history = sample_delivery_success_history()  # Returns integer
        urgency_level = sample_urgency_level()  # Returns integer

        # Convert location_type string to integer (0: Rural, 1: Urban)
        location_type_int = 0 if location_type == 'Rural' else 1

        state = {
            'num_students': num_students,
            'textbooks_kiny': t_kiny,
            'textbooks_eng': t_eng,
            'textbooks_math': t_math,
            'guides_kiny': int(g_kiny),
            'guides_eng': int(g_eng),
            'guides_math': int(g_math),
            'quality_kiny': int(q_kiny),
            'quality_eng': int(q_eng),
            'quality_math': int(q_math),
            'grant_usage': grant_usage,
            'time_since_last_delivery': time_since_last_delivery,
            'delivery_success_history': delivery_success_history,  # Integer
            'urgency_level': urgency_level,  # Integer
            'infrastructure_rating': infrastructure_rating,
            'location_type': location_type_int  # Integer
        }
        # print(f"Debug - State: {state}")  # Debug print to check values
        self._validate_state(state)
        return state

    def _validate_state(self, state):
        if not numStudentState().verify(state['num_students']):
            raise ValueError(f"Invalid num_students: {state['num_students']}")
        if not verify_textbooks_available(state['textbooks_kiny'], state['textbooks_eng'], state['textbooks_math']):
            raise ValueError(f"Invalid textbooks: {state['textbooks_kiny']}, {state['textbooks_eng']}, {state['textbooks_math']}")
        if not verify_teacher_guide_availability(state['guides_kiny'], state['guides_eng'], state['guides_math']):
            raise ValueError(f"Invalid guides: {state['guides_kiny']}, {state['guides_eng']}, {state['guides_math']}")
        if not verify_textbook_quality_score(state['quality_kiny'], state['quality_eng'], state['quality_math']):
            raise ValueError(f"Invalid quality scores: {state['quality_kiny']}, {state['quality_eng']}, {state['quality_math']}")
        if not verify_grant_usage(state['grant_usage']):
            raise ValueError(f"Invalid grant_usage: {state['grant_usage']}")
        if not verify_time_since_last_delivery(state['time_since_last_delivery']):
            raise ValueError(f"Invalid time_since_last_delivery: {state['time_since_last_delivery']}")
        if not verify_delivery_success_history(delivery_success_to_str(state['delivery_success_history'])):  # Convert integer to string
            raise ValueError(f"Invalid delivery_success_history: {state['delivery_success_history']}")
        if not verify_urgency_level(urgency_to_str(state['urgency_level'])):  # Convert integer to string
            raise ValueError(f"Invalid urgency_level: {state['urgency_level']}")
        if not verify_infrastructure_rating(state['infrastructure_rating']):
            raise ValueError(f"Invalid infrastructure_rating: {state['infrastructure_rating']}")
        if not verify_location_type(state['location_type']):  # Now expects integer
            raise ValueError(f"Invalid location_type: {state['location_type']}")

    def _format_observation(self, state):
        """Convert state dict to observation dict with correct shapes for Gym spaces."""
        return {
            'num_students': np.array([state['num_students']], dtype=np.int32),
            'textbooks_kiny': np.array([state['textbooks_kiny']], dtype=np.int32),
            'textbooks_eng': np.array([state['textbooks_eng']], dtype=np.int32),
            'textbooks_math': np.array([state['textbooks_math']], dtype=np.int32),
            'guides_kiny': np.array([state['guides_kiny']], dtype=np.int32),
            'guides_eng': np.array([state['guides_eng']], dtype=np.int32),
            'guides_math': np.array([state['guides_math']], dtype=np.int32),
            'quality_kiny': np.array([state['quality_kiny']], dtype=np.int32),
            'quality_eng': np.array([state['quality_eng']], dtype=np.int32),
            'quality_math': np.array([state['quality_math']], dtype=np.int32),
            'grant_usage': np.array([state['grant_usage']], dtype=np.int32),
            'time_since_last_delivery': np.array([state['time_since_last_delivery']], dtype=np.int32),
            'delivery_success_history': state['delivery_success_history'],
            'urgency_level': state['urgency_level'],
            'infrastructure_rating': np.array([state['infrastructure_rating']], dtype=np.int32),
            'location_type': state['location_type']
        }

    def reset(self, seed=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.num_teachers = np.clip(np.round(np.random.normal(2, 0.5)), 1, 3).astype(int)
        self.state = self._initialize_state()
        self.truck_angle = 0
        self.truck_frame = 0
        self.blink_frame = 0
        self.paused = False
        self.last_action = {'reward': 0}  # Initialize last_action as a dictionary
        if self.render_mode == "human" and self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            self.clock = pygame.time.Clock()
            sprite_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sprites')
            self.sprites['student'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'student.png')).convert_alpha(), (32, 32))
            self.sprites['textbook_kiny'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'textbook_kiny.png')).convert_alpha(), (16, 16))
            self.sprites['textbook_eng'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'textbook_eng.png')).convert_alpha(), (16, 16))
            self.sprites['textbook_math'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'textbook_math.png')).convert_alpha(), (16, 16))
            self.sprites['teacher'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'teacher.png')).convert_alpha(), (80, 80))
            self.sprites['guide_kiny'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'guide_kiny.png')).convert_alpha(), (16, 16))
            self.sprites['guide_eng'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'guide_eng.png')).convert_alpha(), (16, 16))
            self.sprites['guide_math'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'guide_math.png')).convert_alpha(), (16, 16))
            self.sprites['truck_frame1'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'truck_frame1.png')).convert_alpha(), (48, 32))
            self.sprites['truck_frame2'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'truck_frame2.png')).convert_alpha(), (48, 32))
            self.sprites['truck_frame3'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'truck_frame3.png')).convert_alpha(), (48, 32))
            try:
                self.sprites['background'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'background.png')).convert(), (800, 600))
            except pygame.error:
                print("Warning: background.png not found or invalid. Using gradient background.")
                self.sprites['background'] = pygame.Surface((800, 600))
                for y in range(600):
                    r = int(200 + 55 * math.sin(y * 0.02))
                    r = max(0, min(255, r))  # Clamp r to valid range
                    self.sprites['background'].fill((r, 150, 200), (0, y, 800, 1))
        if self.screen and self.render_mode == "human":
            self.screen.fill((255, 255, 255))
            pygame.display.flip()
        return self._format_observation(self.state), {}

    def step(self, action):
        # Convert discrete action index to action tuple if needed
        if isinstance(action, (int, np.integer)):
            action = self.decode_action(action)
        self.last_action = {'action': action, 'reward': 0}  # Store action and initialize reward
        self.current_step += 1
        new_state, reward = step(self.state, action, self.num_teachers)
        self.state = new_state
        self.last_action['reward'] = reward  # Update reward from step
        self.truck_angle = (self.truck_angle + 0.05) % (2 * math.pi)
        self.truck_frame = (self.truck_frame + 1) % 3
        self.blink_frame = (self.blink_frame + 1) % 30
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        info = {}
        return self._format_observation(self.state), reward, terminated, truncated, info

    def _is_terminated(self):
        T_avg = sum([self.state[f'textbooks_{subject}'] for subject in ['kiny', 'eng', 'math']]) / (3 * self.state['num_students'])
        return T_avg >= 0.9 and urgency_to_int(self.state['urgency_level']) == 0  # Using integer 0 for 'Low'

    def render(self):
        if self.render_mode != "human":
            return
        if self.screen is None:
            pygame.init()
            self.screen = pygame.display.set_mode((800, 600))
            self.clock = pygame.time.Clock()
            sprite_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'sprites')
            self.sprites['student'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'student.png')).convert_alpha(), (32, 32))
            self.sprites['textbook_kiny'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'textbook_kiny.png')).convert_alpha(), (16, 16))
            self.sprites['textbook_eng'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'textbook_eng.png')).convert_alpha(), (16, 16))
            self.sprites['textbook_math'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'textbook_math.png')).convert_alpha(), (16, 16))
            self.sprites['teacher'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'teacher.png')).convert_alpha(), (80, 80))
            self.sprites['guide_kiny'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'guide_kiny.png')).convert_alpha(), (16, 16))
            self.sprites['guide_eng'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'guide_eng.png')).convert_alpha(), (16, 16))
            self.sprites['guide_math'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'guide_math.png')).convert_alpha(), (16, 16))
            self.sprites['truck_frame1'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'truck_frame1.png')).convert_alpha(), (48, 32))
            self.sprites['truck_frame2'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'truck_frame2.png')).convert_alpha(), (48, 32))
            self.sprites['truck_frame3'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'truck_frame3.png')).convert_alpha(), (48, 32))
            try:
                self.sprites['background'] = pygame.transform.scale(pygame.image.load(os.path.join(sprite_dir, 'background.png')).convert(), (800, 600))
            except pygame.error:
                print("Warning: background.png not found or invalid. Using gradient background.")
                self.sprites['background'] = pygame.Surface((800, 600))
                for y in range(600):
                    r = int(200 + 55 * math.sin(y * 0.02))
                    r = max(0, min(255, r))  # Clamp r to valid range
                    self.sprites['background'].fill((r, 150, 200), (0, y, 800, 1))

        # Handle events for interactivity
        for event in pygame.event.get():
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_p:
                    self.paused = not self.paused
            if event.type == pygame.QUIT:
                self.close()

        if self.paused:
            return

        # Set background based on urgency (static color)
        urgency_str = urgency_to_str(self.state['urgency_level'])
        urgency_colors = {'Low': (0, 128, 0), 'Medium': (255, 165, 0), 'High': (255, 0, 0)}  # String keys
        base_color = urgency_colors.get(urgency_str, (200, 150, 200))
        background = self.sprites['background'].copy()
        background.fill(base_color)  # Static color instead of gradient
        self.screen.blit(background, (0, 0))

        # Initialize anti-aliased font
        font = pygame.font.SysFont('Arial', 10)  # Smaller font to fit more text

        # Display all state variables higher up
        y_offset = 400
        for key, value in self.state.items():
            state_text = font.render(f"{key}: {value}", True, (255, 255, 255))
            state_surface = pygame.Surface((state_text.get_width() + 5, state_text.get_height() + 5))
            state_surface.set_alpha(200)
            state_surface.fill((0, 0, 0))
            state_surface.blit(state_text, (2, 2))
            self.screen.blit(state_surface, (10, y_offset))
            y_offset += 12  # Tighter spacing to fit all states

        # Center coordinates
        center_x, center_y = 400, 300
        vertical_offset = -60

        # Draw Students header and images with blinking
        students_text = font.render('Students', True, (255, 255, 255))
        students_text_surface = pygame.Surface((students_text.get_width() + 10, students_text.get_height() + 10))
        students_text_surface.set_alpha(180)
        students_text_surface.fill((0, 0, 0))
        students_text_surface.blit(students_text, (5, 5))
        self.screen.blit(students_text_surface, (center_x - 180, center_y + vertical_offset - 40))
        max_students = min(self.state['num_students'] // 5, 5)
        for i in range(max_students):
            x = center_x - 180 + i * 60
            y = center_y + vertical_offset
            student = self.sprites['student'].copy()
            if self.blink_frame < 15:
                student.fill((0, 255, 255), special_flags=pygame.BLEND_RGB_ADD)  # Cyan when blinking
            self.screen.blit(student, (x, y))
            # Tooltip on hover
            if pygame.mouse.get_pos()[0] in range(x, x + 32) and pygame.mouse.get_pos()[1] in range(y, y + 32):
                tooltip = font.render(f"Students: {self.state['num_students']}", True, (255, 255, 255))
                tooltip_surface = pygame.Surface((tooltip.get_width() + 10, tooltip.get_height() + 10))
                tooltip_surface.set_alpha(200)
                tooltip_surface.fill((0, 0, 0))
                tooltip_surface.blit(tooltip, (5, 5))
                self.screen.blit(tooltip_surface, (x, y - 40))

        # Shift textbooks section down
        vertical_offset += 70  # Increased from 50 to 70

        # Draw Textbooks header and images
        textbooks_text = font.render('Textbooks', True, (255, 255, 255))
        textbooks_text_surface = pygame.Surface((textbooks_text.get_width() + 10, textbooks_text.get_height() + 10))
        textbooks_text_surface.set_alpha(180)
        textbooks_text_surface.fill((0, 0, 0))
        textbooks_text_surface.blit(textbooks_text, (5, 5))
        self.screen.blit(textbooks_text_surface, (center_x - 80, center_y + vertical_offset - 40))
        subjects = ['kiny', 'eng', 'math']
        colors = [(0, 100, 0), (0, 0, 100), (100, 0, 0)]
        for i, subject in enumerate(subjects):
            count = min(self.state[f'textbooks_{subject}'], 5)
            x = center_x - 80 + i * 30
            for j in range(count):
                textbook = self.sprites[f'textbook_{subject}'].copy()
                textbook.fill(colors[i], special_flags=pygame.BLEND_RGB_ADD)
                self.screen.blit(textbook, (x, center_y + vertical_offset + j * 30))
                if pygame.mouse.get_pos()[0] in range(x, x + 16) and pygame.mouse.get_pos()[1] in range(center_y + vertical_offset + j * 30, center_y + vertical_offset + j * 30 + 16):
                    tooltip = font.render(f"{subject.capitalize()}: {self.state[f'textbooks_{subject}']}", True, (255, 255, 255))
                    tooltip_surface = pygame.Surface((tooltip.get_width() + 10, tooltip.get_height() + 10))
                    tooltip_surface.set_alpha(200)
                    tooltip_surface.fill((0, 0, 0))
                    tooltip_surface.blit(tooltip, (5, 5))
                    self.screen.blit(tooltip_surface, (x, center_y + vertical_offset + j * 30 - 40))

        # Shift teacher and guides section down
        vertical_offset += 70  # Increased from previous value

        # Draw Teacher header and image
        teacher_text = font.render('Teacher', True, (255, 255, 255))
        teacher_text_surface = pygame.Surface((teacher_text.get_width() + 10, teacher_text.get_height() + 10))
        teacher_text_surface.set_alpha(180)
        teacher_text_surface.fill((0, 0, 0))
        teacher_text_surface.blit(teacher_text, (5, 5))
        self.screen.blit(teacher_text_surface, (center_x + 60, center_y + vertical_offset - 40))
        teacher = pygame.transform.scale(self.sprites['teacher'], (80, 80))
        teacher.fill((255, 255, 0), special_flags=pygame.BLEND_RGB_ADD)
        self.screen.blit(teacher, (center_x + 60, center_y + vertical_offset))

        # Draw Guides header and images
        guides_text = font.render('Guides', True, (255, 255, 255))
        guides_text_surface = pygame.Surface((guides_text.get_width() + 10, guides_text.get_height() + 10))
        guides_text_surface.set_alpha(180)
        guides_text_surface.fill((0, 0, 0))
        guides_text_surface.blit(guides_text, (5, 5))
        self.screen.blit(guides_text_surface, (center_x + 160, center_y + vertical_offset - 40))
        guide_colors = [(255, 255, 200), (255, 200, 255), (200, 200, 200)]
        for i, subject in enumerate(subjects):
            count = min(self.state[f'guides_{subject}'], 3)
            x = center_x + 160 + i * 30
            for j in range(count):
                guide = self.sprites[f'guide_{subject}'].copy()
                guide.fill(guide_colors[i], special_flags=pygame.BLEND_RGB_ADD)
                if i == 2:
                    guide_outline = guide.copy()
                    guide_outline.fill((0, 0, 0), special_flags=pygame.BLEND_RGB_ADD)
                    self.screen.blit(guide_outline, (x - 1, center_y + vertical_offset + j * 30 - 1))
                    self.screen.blit(guide_outline, (x + 1, center_y + vertical_offset + j * 30 + 1))
                self.screen.blit(guide, (x, center_y + vertical_offset + j * 30))
                if pygame.mouse.get_pos()[0] in range(x, x + 16) and pygame.mouse.get_pos()[1] in range(center_y + vertical_offset + j * 30, center_y + vertical_offset + j * 30 + 16):
                    tooltip = font.render(f"{subject.capitalize()}: {self.state[f'guides_{subject}']}", True, (255, 255, 255))
                    tooltip_surface = pygame.Surface((tooltip.get_width() + 10, tooltip.get_height() + 10))
                    tooltip_surface.set_alpha(200)
                    tooltip_surface.fill((0, 0, 0))
                    tooltip_surface.blit(tooltip, (5, 5))
                    self.screen.blit(tooltip_surface, (x, center_y + vertical_offset + j * 30 - 40))

        # Draw truck with circular movement
        radius = 300
        center_x, center_y = 400, 300
        truck_x = int(center_x + radius * math.cos(self.truck_angle))
        truck_y = int(center_y + radius * math.sin(self.truck_angle))
        truck_frames = ['truck_frame1', 'truck_frame2', 'truck_frame3']
        # Truck color based on delivery success history
        delivery_success_str = delivery_success_to_str(self.state['delivery_success_history'])
        truck_color = {'Low': (255, 50, 50), 'Medium': (255, 200, 50), 'High': (50, 255, 50)}  # String keys
        truck = self.sprites[truck_frames[self.truck_frame]].convert_alpha()
        truck.fill(truck_color.get(delivery_success_str, (128, 128, 128)), special_flags=pygame.BLEND_RGB_ADD)
        self.screen.blit(truck, (truck_x - 24, truck_y - 16))

        # Real-time feedback status bar
        status_font = pygame.font.SysFont('Arial', 16)
        T_avg = sum([self.state[f'textbooks_{subject}'] for subject in ['kiny', 'eng', 'math']]) / (3 * self.state['num_students'])
        # Status bar
        urgency_str = urgency_to_str(self.state['urgency_level'])
        status_text = status_font.render(f"Step: {self.current_step} | Urgency: {urgency_str} | Reward: {self.last_action['reward']:.2f} | T_avg: {T_avg:.2f}", True, (255, 255, 255))
        status_surface = pygame.Surface((status_text.get_width() + 10, status_text.get_height() + 10))
        status_surface.set_alpha(200)
        status_surface.fill((0, 0, 0))
        status_surface.blit(status_text, (5, 5))
        self.screen.blit(status_surface, (10, 10))

        pygame.display.flip()
        self.clock.tick(30)

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None

if __name__ == "__main__":
    env = P3Environment(max_steps=100, render_mode="human")
    state, _ = env.reset()
    env.render()
    for _ in range(10):
        action = env.sample()
        state, reward, terminated, truncated, _ = env.step(action)
        print(f"Reward: {reward}, T_avg: {sum([state[f'textbooks_{subject}'] for subject in ['kiny', 'eng', 'math']]) / (3 * state['num_students']):.2f}, Urgency: {urgency_to_str(state['urgency_level'])}")
        env.render()
    env.close()