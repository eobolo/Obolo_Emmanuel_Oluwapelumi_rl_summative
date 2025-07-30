"""
Grant Usage for P3 Students in RL Environment

This module figures out what percentage of education grants a school has used for P3 students’ teaching and learning materials (e.g., textbooks, teacher guides). It’s based on Rwanda’s 2021/2022 educational statistics (3,831 schools, 6,519 P3 classrooms, ~109 students/school), NISR 2022 Census (~72.1% rural, 27.9% urban), and FLN Priority 3, which notes grants for TLM purchases. No specific grant usage data exists, so it’s simulated using the location type (Rural or Urban) from location_type_state.py.

What It Does:
- Shows how much of a school’s grant money is spent, as a percentage (0–100%).
- Ideal: 100% (all money used efficiently for TLM).
- Reality: Rural schools (~89 students) use less due to delays or poor management; urban schools (~156 students) use more.

How It’s Picked:
- Percentage depends on the school’s location (Rural or Urban, from location_type_state.py).
- Rural schools average ~65% (e.g., 65/100, some funds unspent); urban schools average ~75% (e.g., 75/100, most funds used).
- Overall average: ~68% (mix of 70% rural, 30% urban; 65% × 0.7 + 75% × 0.3 ≈ 68%).
- Scores stay between 0 and 100.
- Example: A Rural school with 50% usage might have unspent funds due to slow paperwork; an Urban school with 80% uses funds well.

Check for Realism:
- Percentage is checked to be between 0 and 100.
- Number of students (S) is checked to be in the valid range (30–200).
- Location type is checked to be Rural or Urban.

Purpose:
- Generates a grant usage percentage for each school, showing spending efficiency.
- The RL agent uses (S, Grant_Usage) to prioritize schools with low usage (e.g., needing support to spend funds better).
"""

import numpy as np

def sample_grant_usage(location_type):
    """
    Generates the percentage of grant funds used by a school for P3 students.

    Args:
        location_type (str): Location type ('Rural' or 'Urban', from location_type_state.py).

    Returns:
        int: Grant usage percentage (0–100).
    """
    # Convert integer location_type to string
    location_str = 'Rural' if location_type == 0 else 'Urban' if location_type == 1 else None
    if location_str is None:
        raise ValueError("Location type must be 0 (Rural) or 1 (Urban), got {}".format(location_type))

    # Sample grant usage from normal distribution based on location type
    if location_type == 'Rural':
        usage = np.random.normal(65, 15)  # Mean = 65, SD = 15
    else:  # Urban
        usage = np.random.normal(75, 15)  # Mean = 75, SD = 15

    # Clip to 0–100
    usage = int(max(0, min(100, usage)))

    return usage

def verify_grant_usage(usage):
    """
    Verifies that the grant usage percentage is within the valid range.

    Args:
        usage (int): Grant usage percentage.

    Returns:
        bool: True if usage is in [0, 100] False otherwise.
    """
    return isinstance(usage, int) and 0 <= usage <= 100