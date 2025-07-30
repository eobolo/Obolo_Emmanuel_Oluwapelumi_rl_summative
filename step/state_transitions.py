import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from reward.reward_function import calculate_reward
from state.num_of_students_state import numStudentState
from state.textbook_available_state import verify_textbooks_available
from state.teacher_guide_availability_state import verify_teacher_guide_availability
from state.textbook_quality_score import verify_textbook_quality_score
from state.grant_usage_state import verify_grant_usage
from state.time_since_last_delivery_state import verify_time_since_last_delivery
from state.delivery_success_history_state import verify_delivery_success_history
from state.urgency_level_state import verify_urgency_level
from state.infrastructure_rating_state import verify_infrastructure_rating
from state.location_type_state import verify_location_type

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

def step(state, action, num_teachers):
    """
    Updates the state based on the action and returns the new state and reward.

    Args:
        state (dict): Current state with keys: num_students, textbooks_kiny, textbooks_eng, textbooks_math,
                      guides_kiny, guides_eng, guides_math, quality_kiny, quality_eng, quality_math,
                      grant_usage, time_since_last_delivery, delivery_success_history, urgency_level,
                      infrastructure_rating, location_type.
        action (tuple): (D_kiny, D_eng, D_math, DG_kiny, DG_eng, DG_math), non-negative integers.
        num_teachers (int): Number of P3 teachers (1–3).

    Returns:
        tuple: (new_state, reward)
            - new_state (dict): Updated state with same keys.
            - reward (float): Reward from reward_function.calculate_reward.
    """
    # Validate inputs
    num_students_instance = numStudentState()
    if not num_students_instance.verify(state['num_students']):
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
    delivery_history = delivery_success_to_str(state['delivery_success_history'])
    if not verify_delivery_success_history(delivery_history):
        raise ValueError(f"Invalid delivery_success_history: {state['delivery_success_history']}")
    urgency = urgency_to_str(state['urgency_level'])
    if not verify_urgency_level(urgency):
        raise ValueError(f"Invalid urgency_level: {state['urgency_level']}")
    if not verify_infrastructure_rating(state['infrastructure_rating']):
        raise ValueError(f"Invalid infrastructure_rating: {state['infrastructure_rating']}")
    if not verify_location_type(state['location_type']):
        raise ValueError(f"Invalid location_type: {state['location_type']}")

    # Create new state
    new_state = state.copy()

    # Extract action components
    D_kiny, D_eng, D_math, DG_kiny, DG_eng, DG_math = action
    total_delivered = sum([D_kiny, D_eng, D_math, DG_kiny, DG_eng, DG_math])

    # Determine delivery success
    delivery_success = delivery_success_to_str(state['delivery_success_history'])
    if delivery_success == 'Low':
        probabilities = [0.2, 0.5, 0.3]  # Full, partial, fail
    elif delivery_success == 'Medium':
        probabilities = [0.5, 0.3, 0.2]
    else:  # High
        probabilities = [0.8, 0.15, 0.05]
    outcome = np.random.choice(['full', 'partial', 'fail'], p=probabilities)
    success_factor = 1.0 if outcome == 'full' else 0.5 if outcome == 'partial' else 0.0

    # Dynamic capping based on num_students and num_teachers
    max_textbook_delivery = min(int(0.25 * state['num_students']), 200)  # 25% of students, max 200
    max_guide_delivery = min(int(0.10 * num_teachers), 3)  # 10% of teachers, max 3

    # Update Textbooks Available
    for subject, D in [('kiny', D_kiny), ('eng', D_eng), ('math', D_math)]:
        delivered = int(min(D * success_factor, max_textbook_delivery))
        new_state[f'textbooks_{subject}'] = int(state[f'textbooks_{subject}'] + delivered)

    # Update Teacher Guide Availability
    for subject, DG in [('kiny', DG_kiny), ('eng', DG_eng), ('math', DG_math)]:
        delivered = int(min(DG * success_factor, max_guide_delivery))
        new_state[f'guides_{subject}'] = int(state[f'guides_{subject}'] + delivered)

    # Update Textbook Quality Score (capped at 100)
    for subject, T_old, D, Q_old in [
        ('kiny', state['textbooks_kiny'], D_kiny, state['quality_kiny']),
        ('eng', state['textbooks_eng'], D_eng, state['quality_eng']),
        ('math', state['textbooks_math'], D_math, state['quality_math'])
    ]:
        T_new = new_state[f'textbooks_{subject}']
        delivered = int(min(D * success_factor, max_textbook_delivery))
        if delivered > 0 and T_new > 0:
            new_state[f'quality_{subject}'] = int(min((T_old * Q_old + delivered * 80) / T_new, 100))
        else:
            new_state[f'quality_{subject}'] = int(Q_old)

    # Update Grant Usage and Infrastructure Rating for Flag/Hold
    if total_delivered == 0:  # Hold/Flag
        if np.random.random() < 0.5:  # 50% chance of Flag
            new_state['grant_usage'] = int(min(state['grant_usage'] + 10, 100))  # +10% for Flag
            new_state['infrastructure_rating'] = int(min(state['infrastructure_rating'] + 5, 100))  # +5 for Flag
        # Else Hold: no change
    else:
        new_state['grant_usage'] = int(state['grant_usage'])
        new_state['infrastructure_rating'] = int(state['infrastructure_rating'])

    # Update Time Since Last Delivery
    if total_delivered > 0 and success_factor > 0:  # Successful delivery (full or partial)
        new_state['time_since_last_delivery'] = int(0)
    else:
        new_state['time_since_last_delivery'] = int(min(state['time_since_last_delivery'] + 30, np.inf))

    # Update Delivery Success History
    success_rate = {'Low': 20, 'Medium': 50, 'High': 80}[delivery_success_to_str(state['delivery_success_history'])]  # Hidden numerical rate
    if total_delivered > 0:
        if outcome == 'full':
            success_rate = min(success_rate + 10, 100)
        elif outcome == 'fail':
            success_rate = max(success_rate - 5, 0)
    # Always store as int for consistency
    if success_rate < 30:
        new_state['delivery_success_history'] = 0
    elif success_rate <= 70:
        new_state['delivery_success_history'] = 1
    else:
        new_state['delivery_success_history'] = 2

    # Update Urgency Level
    T_avg = sum([new_state[f'textbooks_{subject}'] for subject in ['kiny', 'eng', 'math']]) / (3 * state['num_students'])
    # Always store as int for consistency
    if T_avg < 0.5:
        new_state['urgency_level'] = 2
    elif T_avg <= 0.9:
        new_state['urgency_level'] = 1
    else:
        new_state['urgency_level'] = 0

    # Static states
    new_state['num_students'] = int(state['num_students'])
    new_state['location_type'] = state['location_type']

    # Calculate reward with both old and new states
    reward = calculate_reward(state, new_state, action, num_teachers)

    return new_state, reward