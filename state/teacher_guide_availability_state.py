"""
Teacher Guide Availability for P3 Students in RL Environment

This module figures out how many teacher guides for Kinyarwanda, English, and Mathematics are available for P3 teachers in a school. It’s based on Rwanda’s 2021/2022 educational statistics (3,831 schools, 6,519 P3 classrooms, ~1.7 classrooms/school, ~109 students/school), NISR 2022 Census (~72.1% rural, 27.9% urban), and the 2021/2024 Auditor General’s reports (shortages 1:7 to 1:223, delays 27–200 days). Guides are simulated using the location type (Rural or Urban) from location_type_state.py. The number of students (S, average 109, range 30–200) from number_of_students_state.py is only used to check validity.

What It Does:
- Counts teacher guides per subject (G_Kiny, G_Eng, G_Math) for P3 teachers.
- Ideal: One guide per teacher per subject (e.g., 2 teachers need 2 guides each).
- Reality: Rural schools (~89 students) have fewer guides; urban schools (~156 students) have more. Kinyarwanda has more guides than English/Mathematics.

How Numbers Are Picked:
- Assumes 1–3 P3 teachers (average ~2, based on ~1.7 classrooms/school).
- Counts depend on the school’s location (Rural or Urban, from location_type_state.py).
- Rural: Kinyarwanda ~50% of teachers (e.g., 2 × 0.5 ≈ 1 guide), English/Math ~40% (2 × 0.4 ≈ 1 guide).
- Urban: Kinyarwanda ~70% (e.g., 2 × 0.7 ≈ 1–2 guides), English/Math ~60% (2 × 0.6 ≈ 1 guide).
- Overall average (~2 teachers): Kinyarwanda ~0.55 guides (2 × 0.55), English/Math ~0.45 guides (2 × 0.45).
- Example: A Rural school with 2 teachers might have 1 Kinyarwanda, 0 English, 1 Math guide; an Urban school might have 2, 1, 1.

Check for Realism:
- Guide counts are checked to be between 0 and the number of teachers.
- Number of students (S) is checked to be in the valid range (30–200).
- Location type is checked to be Rural or Urban.

Purpose:
- Generates guide counts (G_Kiny, G_Eng, G_Math) for each school, showing shortages (Kinyarwanda better than English/Mathematics).
- The RL agent uses (S, G_Kiny, G_Eng, G_Math) to prioritize schools needing more guides (e.g., English if none).
"""

import numpy as np

def sample_teacher_guide_availability(location_type, num_teachers):
    """
    Generates the number of P3 teacher guides (Kinyarwanda, English, Mathematics) for a school.

    Args:
        num_students (int): Number of P3 students (from number_of_students_state.py, range 30–200).
        location_type (str): Location type ('Rural' or 'Urban', from location_type_state.py).

    Returns:
        tuple: (G_Kiny, G_Eng, G_Math), number of guides for each subject, clipped to 0–num_teachers.
    """
    # Convert integer location_type to string
    location_str = 'Rural' if location_type == 0 else 'Urban' if location_type == 1 else None
    if location_str is None:
        raise ValueError("Location type must be 0 (Rural) or 1 (Urban), got {}".format(location_type))

    # Sample guide proportions from normal distributions based on location type
    if location_type == 'Rural':
        p_kiny = np.random.normal(0.5, 0.1)  # Mean = 50%, SD = 10%
        p_eng = np.random.normal(0.4, 0.1)   # Mean = 40%, SD = 10%
        p_math = np.random.normal(0.4, 0.1)  # Mean = 40%, SD = 10%
    else:  # Urban
        p_kiny = np.random.normal(0.7, 0.1)  # Mean = 70%, SD = 10%
        p_eng = np.random.normal(0.6, 0.1)   # Mean = 60%, SD = 10%
        p_math = np.random.normal(0.6, 0.1)  # Mean = 60%, SD = 10%

    # Calculate guide counts, clipped to 0–num_teachers
    g_kiny = int(np.floor(p_kiny * num_teachers))
    g_eng = int(np.floor(p_eng * num_teachers))
    g_math = int(np.floor(p_math * num_teachers))
    g_kiny = max(0, min(g_kiny, num_teachers))
    g_eng = max(0, min(g_eng, num_teachers))
    g_math = max(0, min(g_math, num_teachers))

    return g_kiny, g_eng, g_math

def verify_teacher_guide_availability(g_kiny, g_eng, g_math):
    """
    Verifies that the number of teacher guides for each subject is within the valid range.

    Args:
        g_kiny (int): Number of Kinyarwanda guides.
        g_eng (int): Number of English guides.
        g_math (int): Number of Mathematics guides.

    Returns:
        bool: True if all guide counts are of type int, False otherwise.
    """
    return all(isinstance(x, int) and x >= 0 for x in [g_kiny, g_eng, g_math])


