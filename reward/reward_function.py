"""
Reward Function for P3 RL Environment

This module defines the reward function for the RL environment, evaluating actions to maximize textbook effectiveness and equitable access for P3 schools in Rwanda. It’s based on Rwanda’s 2021/2022 educational statistics (3,831 schools, 6,519 P3 classrooms, ~109 students/school), NISR 2022 Census (~72.1% rural, 27.9% urban), FLN Priority 3 (TLM emphasis), and Auditor General’s reports (shortages 1:7 to 1:223, delays 27–200 days).

---

What It Does:
- Rewards actions that reduce textbook/guide shortages, improve quality, prioritize long-delayed or low-success delivery schools.
- Penalizes over-delivery, waste on unprepared schools, and ignoring high urgency.
- Applies a rural bonus for equitable access.

---

Reward Components:
- Positive: Textbook shortage reduction (higher for English/Math), guide shortage reduction, quality improvement, long time since last delivery (>100 days), low delivery success history (prioritize underserved schools).
- Negative: Over-delivery (textbooks > students, guides > teachers), deliveries to unprepared schools (low Grant Usage or Infrastructure Rating), ignoring high urgency.
- Rural Bonus: 1.2x multiplier for rural schools.

---

Purpose:
- Guides the RL agent to prioritize schools with shortages, ensure effective resource use, avoid waste, and favor rural, urgent, long-delayed, or low-success schools.

---

Notes:
- Uses all 10 states (Number of Students, Textbooks Available, Teacher Guide Availability, Textbook Quality Score, Grant Usage, Time Since Last Delivery, Delivery Success History, Urgency Level, Infrastructure Rating, Location Type).
- Assumes new textbooks/guides have quality ~80.
- Hold/Flag treated as one action (no Grant Usage increase until state transitions defined).
- Time Since Last Delivery uses uniform 0–365 days (REB annual cycle).
- Delivery Success History is categorical ('Low' <30%, 'Medium' 30–70%, 'High' >70%).
"""

import numpy as np

def calculate_reward(old_state, new_state, action, num_teachers):
    """
    Calculates the reward for an action given the old and new states.

    Args:
        old_state (dict): State before the transition.
        new_state (dict): State after the transition.
        action (np.ndarray): (D_kiny, D_eng, D_math, DG_kiny, DG_eng, DG_math), non-negative integers.
        num_teachers (int): Number of P3 teachers (1–3, from Teacher Guide Availability).

    Returns:
        float: Total reward, combining positive rewards and penalties.
    """
    # Extract state and action components
    S = new_state['num_students']
    T_kiny_old = old_state.get('textbooks_kiny', 0)
    T_eng_old = old_state.get('textbooks_eng', 0)
    T_math_old = old_state.get('textbooks_math', 0)
    T_kiny_new, T_eng_new, T_math_new = new_state['textbooks_kiny'], new_state['textbooks_eng'], new_state['textbooks_math']
    G_kiny_old, G_eng_old, G_math_old = old_state.get('guides_kiny', 0), old_state.get('guides_eng', 0), old_state.get('guides_math', 0)
    G_kiny_new, G_eng_new, G_math_new = new_state['guides_kiny'], new_state['guides_eng'], new_state['guides_math']
    Q_kiny_old, Q_eng_old, Q_math_old = old_state.get('quality_kiny', 50), old_state.get('quality_eng', 50), old_state.get('quality_math', 50)
    Q_kiny_new, Q_eng_new, Q_math_new = new_state['quality_kiny'], new_state['quality_eng'], new_state['quality_math']
    grant_usage = new_state['grant_usage']
    time_delivery_old = old_state.get('time_since_last_delivery', 0)
    time_delivery_new = new_state['time_since_last_delivery']
    delivery_success = new_state['delivery_success_history']
    urgency_level = new_state['urgency_level']
    infrastructure = new_state['infrastructure_rating']
    location_type = new_state['location_type']
    D_kiny, D_eng, D_math, DG_kiny, DG_eng, DG_math = action

    # Initialize reward and penalty
    reward = 0.0
    penalty = 0.0

    # Textbook shortage reduction and undersupply/oversupply penalties
    for subject, T_old, T_new, D, w1 in [
        ('Kinyarwanda', T_kiny_old, T_kiny_new, D_kiny, 10),
        ('English', T_eng_old, T_eng_new, D_eng, 15),
        ('Mathematics', T_math_old, T_math_new, D_math, 15)
    ]:
        if T_new > T_old:  # Reward only for actual delivery
            old_ratio = S / max(T_old, 1) if T_old > 0 else S  # Avoid division by zero, minimum ratio
            new_ratio = S / T_new if T_new > 0 else float('inf')
            reward += w1 * max(0, old_ratio - new_ratio)
        # Undersupply and oversupply penalties based on new state
        undersupply = max(0, (S - T_new) / S if T_new > 0 else 1.0)  # Penalty if no textbooks
        oversupply = max(0, (T_new - S) / S if T_new > 0 else 0.0)
        penalty -= 10 * undersupply
        penalty -= 10 * oversupply

    # Guide shortage reduction and undersupply/oversupply penalties
    for subject, G_old, G_new, DG, w2 in [
        ('Kinyarwanda', G_kiny_old, G_kiny_new, DG_kiny, 20),
        ('English', G_eng_old, G_eng_new, DG_eng, 25),
        ('Mathematics', G_math_old, G_math_new, DG_math, 25)
    ]:
        if G_new > G_old:  # Reward only for actual delivery
            reward += w2 * (DG / num_teachers if num_teachers > 0 else 0)
        # Undersupply and oversupply penalties based on new state
        undersupply = max(0, (num_teachers - G_new) / num_teachers if num_teachers > 0 and G_new > 0 else 1.0)
        oversupply = max(0, (G_new - num_teachers) / num_teachers if num_teachers > 0 and G_new > 0 else 0.0)
        penalty -= 20 * undersupply
        penalty -= 20 * oversupply

    # Quality improvement (based on actual change)
    for subject, T_old, T_new, D, Q_old, Q_new in [
        ('Kinyarwanda', T_kiny_old, T_kiny_new, D_kiny, Q_kiny_old, Q_kiny_new),
        ('English', T_eng_old, T_eng_new, D_eng, Q_eng_old, Q_eng_new),
        ('Mathematics', T_math_old, T_math_new, D_math, Q_math_old, Q_math_new)
    ]:
        if T_new > T_old:  # Only reward if textbooks increased
            delivered_quality = 80 if D > 0 else 0
            new_quality = (T_old * Q_old + min(D, T_new - T_old) * delivered_quality) / T_new if T_new > 0 else Q_old
            reward += 5 * max(0, Q_new - Q_old)  # Use actual new quality
        else:
            reward += 5 * (Q_new - 50)  # Baseline quality

    # Grant usage improvement (placeholder, +5 for Flag)
    if np.all(action == 0):  # Hold/Flag
        if np.random.random() < 0.5:  # 50% chance of Flag
            reward += 5  # Incentive for preparedness

    # Time since last delivery reward/penalty
    total_delivered_intended = sum([D_kiny, D_eng, D_math, DG_kiny, DG_eng, DG_math])
    if time_delivery_new == 0 and total_delivered_intended > 0:  # Successful delivery
        reward += 5 * total_delivered_intended
    elif time_delivery_new > time_delivery_old:  # Failed or delayed delivery
        penalty -= 10 * (time_delivery_new - time_delivery_old)
    elif time_delivery_new > 100:
        penalty -= 10 * (time_delivery_new - 100)

    # Delivery success history reward
    if total_delivered_intended > 0 and delivery_success == 'Low':
        reward += 5 * total_delivered_intended  # Prioritize underserved schools

    # Additional penalties
    if total_delivered_intended > 0 and (grant_usage < 50 or infrastructure < 50):
        penalty -= 15 * total_delivered_intended  # Waste penalty
    if urgency_level != 'High' and total_delivered_intended > 0:
        penalty -= 15  # Ignoring high urgency

    # Rural bonus
    total_reward = (reward + penalty) * (1.2 if location_type == 'Rural' else 1.0)

    return total_reward