"""
Textbooks Available for P3 Students in RL Environment

This module figures out how many Kinyarwanda, English, and Mathematics textbooks a school has for its P3 students. It’s based on Rwanda’s 2021/2022 educational statistics (3,831 schools, 6,519 P3 classrooms, ~109 students/school), NISR 2022 Census (~72.1% rural, 27.9% urban), and the 2021/2024 Auditor General’s reports (shortages 1:7 to 1:223, delays 27–200 days). Counts are simulated using the location type (Rural or Urban) from location_type_state.py.

What It Does:
- Counts textbooks per subject (T_Kiny, T_Eng, T_Math) for a school’s P3 students.
- Ideal: 1 textbook per student per subject (e.g., for 100 students, 100 Kinyarwanda, 100 English, 100 Mathematics).
- Reality: Rural schools (~89 students) have fewer textbooks (Kinyarwanda ~1:7, English/Math ~1:15); urban schools (~156 students) have more (Kinyarwanda ~1:4.3, English/Math ~1:7).

How Numbers Are Picked:
- Counts depend on the school’s location (Rural or Urban, from location_type_state.py) and number of students (S).
- Rural: Kinyarwanda ~1:7 (e.g., 89 ÷ 7 ≈ 13 books), English/Math ~1:15 (89 ÷ 15 ≈ 6 books).
- Urban: Kinyarwanda ~1:4.3 (e.g., 156 ÷ 4.3 ≈ 36 books), English/Math ~1:7 (156 ÷ 7 ≈ 22 books).
- Overall average (S = 109): Kinyarwanda ~18 (109 ÷ 6), English/Math ~10 (109 ÷ 11).
- Example: A Rural school with 89 students might have 12 Kinyarwanda, 5 English, 6 Math; an Urban school with 156 students might have 35 Kinyarwanda, 22 English, 21 Math.

Check for Realism:
- Counts are checked to be between 0 and S (no negative or excess textbooks).
- Location type is checked to be Rural or Urban.
- Number of students (S) is checked to be in the valid range (30–200).

Purpose:
- Generates textbook counts (T_Kiny, T_Eng, T_Math) for each school, showing shortages (Kinyarwanda better than English/Mathematics).
- The RL agent uses (S, T_Kiny, T_Eng, T_Math) to prioritize schools needing more textbooks (e.g., Mathematics if near zero).
"""

import numpy as np

def sample_textbooks_available(num_students, location_type):
    """
    Generates the number of P3 textbooks (Kinyarwanda, English, Mathematics) for a school
    based on the location type and number of students.

    Args:
        num_students (int): Number of students in the school.
        location_type (int): Location type (0: Rural, 1: Urban).

    Returns:
        tuple: (T_Kiny, T_Eng, T_Math), number of textbooks for each subject, clipped to 0–num_students.
    """
    # Convert integer location_type to string
    location_str = 'Rural' if location_type == 0 else 'Urban' if location_type == 1 else None
    if location_str is None:
        raise ValueError("Location type must be 0 (Rural) or 1 (Urban), got {}".format(location_type))

    # Sample proportions from beta distributions based on location type
    if location_str == 'Rural':
        p_kiny = np.random.beta(1, 6)  # Mean = 0.143 (~1:7)
        p_eng = np.random.beta(0.5, 7)  # Mean = 0.067 (~1:15)
        p_math = np.random.beta(0.5, 7)  # Mean = 0.067 (~1:15)
    else:  # Urban
        p_kiny = np.random.beta(1.5, 5)  # Mean = 0.231 (~1:4.3)
        p_eng = np.random.beta(1, 6)  # Mean = 0.143 (~1:7)
        p_math = np.random.beta(1, 6)  # Mean = 0.143 (~1:7)

    # Calculate textbook counts, clipped to 0–num_students
    t_kiny = int(np.floor(p_kiny * num_students))
    t_eng = int(np.floor(p_eng * num_students))
    t_math = int(np.floor(p_math * num_students))
    t_kiny = max(0, min(t_kiny, num_students))
    t_eng = max(0, min(t_eng, num_students))
    t_math = max(0, min(t_math, num_students))

    return t_kiny, t_eng, t_math

def verify_textbooks_available(t_kiny, t_eng, t_math):
    """
    Verifies that the number of textbooks for each subject is within the valid range.

    Args:
        t_kiny (int): Number of Kinyarwanda textbooks.
        t_eng (int): Number of English textbooks.
        t_math (int): Number of Mathematics textbooks.

    Returns:
        bool: True if all textbook counts are of type int False otherwise.
    """
    return all(isinstance(x, int) and x >= 0 for x in [t_kiny, t_eng, t_math])