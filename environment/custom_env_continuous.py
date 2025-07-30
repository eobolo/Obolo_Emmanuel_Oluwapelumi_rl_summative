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

class P3EnvironmentContinuous(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, max_steps=100, render_mode=None):
        super(P3EnvironmentContinuous, self).__init__()
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

        # Define observation space (same as discrete)
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
            'delivery_success_history': spaces.Discrete(3),
            'urgency_level': spaces.Discrete(3),
            'infrastructure_rating': spaces.Box(low=0, high=100, shape=(1,), dtype=np.int32),
            'location_type': spaces.Discrete(2)
        })

        # Continuous action space: [D_kiny, D_eng, D_math, DG_kiny, DG_eng, DG_math]
        # Textbooks: 0-200, Guides: 0-3
        self.action_space = spaces.Box(
            low=np.array([0, 0, 0, 0, 0, 0], dtype=np.float32),
            high=np.array([200, 200, 200, 3, 3, 3], dtype=np.float32),
            shape=(6,),
            dtype=np.float32
        )

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
                    r = max(0, min(255, r))
                    self.sprites['background'].fill((r, 150, 200), (0, y, 800, 1))

    def is_paused(self):
        return self.paused

    def sample(self):
        # Sample a continuous action
        return self.action_space.sample()

    def _initialize_state(self):
        num_students_instance = numStudentState()
        location_type = sample_location_type()
        num_students = int(num_students_instance.sample_num_students())
        t_kiny, t_eng, t_math = sample_textbooks_available(num_students, location_type)
        g_kiny, g_eng, g_math = sample_teacher_guide_availability(location_type, self.num_teachers)
        q_kiny, q_eng, q_math = sample_textbook_quality_score(location_type)
        grant_usage = int(sample_grant_usage(location_type))
        infrastructure_rating = int(sample_infrastructure_rating(location_type))
        time_since_last_delivery = int(sample_time_since_last_delivery())
        delivery_success_history = sample_delivery_success_history()
        urgency_level = sample_urgency_level()
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
            'delivery_success_history': delivery_success_history,
            'urgency_level': urgency_level,
            'infrastructure_rating': infrastructure_rating,
            'location_type': location_type_int
        }
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
        if not verify_delivery_success_history(delivery_success_to_str(state['delivery_success_history'])):
            raise ValueError(f"Invalid delivery_success_history: {state['delivery_success_history']}")
        if not verify_urgency_level(urgency_to_str(state['urgency_level'])):
            raise ValueError(f"Invalid urgency_level: {state['urgency_level']}")
        if not verify_infrastructure_rating(state['infrastructure_rating']):
            raise ValueError(f"Invalid infrastructure_rating: {state['infrastructure_rating']}")
        if not verify_location_type(state['location_type']):
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
        self.last_action = {'reward': 0}
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
                    r = max(0, min(255, r))
                    self.sprites['background'].fill((r, 150, 200), (0, y, 800, 1))
        if self.screen and self.render_mode == "human":
            self.screen.fill((255, 255, 255))
            pygame.display.flip()
        return self._format_observation(self.state), {}

    def step(self, action):
        # Accepts a 6D float vector, rounds/clips to valid delivery amounts
        action = np.array(action, dtype=np.float32)
        # Textbooks: 0-200, Guides: 0-3
        action[:3] = np.clip(np.round(action[:3]), 0, 200)  # Textbooks
        action[3:] = np.clip(np.round(action[3:]), 0, 3)    # Guides
        action_tuple = tuple(int(x) for x in action)
        self.last_action = {'action': action_tuple, 'reward': 0}
        self.current_step += 1
        new_state, reward = step(self.state, action_tuple, self.num_teachers)
        self.state = new_state
        self.last_action['reward'] = reward
        self.truck_angle = (self.truck_angle + 0.05) % (2 * math.pi)
        self.truck_frame = (self.truck_frame + 1) % 3
        self.blink_frame = (self.blink_frame + 1) % 30
        terminated = self._is_terminated()
        truncated = self.current_step >= self.max_steps
        info = {}
        return self._format_observation(self.state), reward, terminated, truncated, info

    def _is_terminated(self):
        T_avg = sum([self.state[f'textbooks_{subject}'] for subject in ['kiny', 'eng', 'math']]) / (3 * self.state['num_students'])
        return T_avg >= 0.9 and urgency_to_int(self.state['urgency_level']) == 0

    def render(self):
        # (Same as discrete version)
        pass  # You can copy the render logic if needed

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
            self.clock = None 