"""
Textbook Quality Score for P3 Students in RL Environment

This module figures out the quality of Kinyarwanda, English, and Mathematics textbooks in a school for P3 students. It’s based on Rwanda’s 2021/2022 educational statistics (3,831 schools, 6,519 P3 classrooms, ~109 students/school), NISR 2022 Census (~72.1% rural, 27.9% urban), and the 2021/2024 Auditor General’s reports (textbook shortages 1:7 to 1:223, delays 27–200 days). Quality is simulated using the location type (Rural or Urban) from location_type_state.py, as no specific quality data exists. The number of students (S, average 109, range 30–200) from number_of_students_state.py is only used to check validity.

What It Does:
- Gives a quality score (0–100) for each textbook (Kinyarwanda, English, Mathematics) based on durability (e.g., not torn) and clarity (e.g., readable, curriculum-aligned).
- Ideal: 100 (perfect condition, curriculum-aligned).
- Reality: Rural schools (~89 students) have poorer quality; urban schools (~156 students) have better quality. Kinyarwanda books are often better (newer, more available).

How It’s Picked:
- Scores depend on the school’s location (Rural or Urban, from location_type_state.py).
- Rural schools: Kinyarwanda ~60 (e.g., moderately worn), English/Math ~50 (e.g., more wear, less clear).
- Urban schools: Kinyarwanda ~70 (e.g., good condition), English/Math ~60 (e.g., slightly worn).
- Overall average: Kinyarwanda ~63 (60 × 0.7 + 70 × 0.3), English/Math ~53 (50 × 0.7 + 60 × 0.3).
- Scores stay between 0 and 100.
- Example: A Rural school might have scores 58 (Kinyarwanda), 47 (English), 52 (Math); an Urban school might have 72, 61, 59.

Check for Realism:
- Scores are checked to be between 0 and 100.
- Number of students (S) is checked to be in the valid range (30–200).
- Location type is checked to be Rural or Urban.

Purpose:
- Generates quality scores (Q_Kiny, Q_Eng, Q_Math) for each school, showing differences (Kinyarwanda better than English/Mathematics).
- The RL agent uses (S, Q_Kiny, Q_Eng, Q_Math) to prioritize schools needing textbook replacements (e.g., low English score).
"""

import numpy as np

def sample_textbook_quality_score(location_type):
    """
    Generates quality scores (0–100) for P3 textbooks (Kinyarwanda, English, Mathematics) for a school.

    Args:
        location_type (str): Location type ('Rural' or 'Urban', from location_type_state.py).

    Returns:
        tuple: (Q_Kiny, Q_Eng, Q_Math), quality scores for each subject, clipped to 0–100.
    """
    # Convert integer location_type to string
    location_str = 'Rural' if location_type == 0 else 'Urban' if location_type == 1 else None
    if location_str is None:
        raise ValueError("Location type must be 0 (Rural) or 1 (Urban), got {}".format(location_type))

    # Sample quality scores from normal distributions based on location type
    if location_type == 'Rural':
        q_kiny = np.random.normal(60, 15)  # Mean = 60, SD = 15
        q_eng = np.random.normal(50, 15)   # Mean = 50, SD = 15
        q_math = np.random.normal(50, 15)  # Mean = 50, SD = 15
    else:  # Urban
        q_kiny = np.random.normal(70, 15)  # Mean = 70, SD = 15
        q_eng = np.random.normal(60, 15)   # Mean = 60, SD = 15
        q_math = np.random.normal(60, 15)  # Mean = 60, SD = 15

    # Clip scores to 0–100
    q_kiny = int(max(0, min(100, q_kiny)))
    q_eng = int(max(0, min(100, q_eng)))
    q_math = int(max(0, min(100, q_math)))

    return q_kiny, q_eng, q_math

def verify_textbook_quality_score(q_kiny, q_eng, q_math):
    """
    Verifies that textbook quality scores are within the valid range.

    Args:
        q_kiny (int): Quality score for Kinyarwanda.
        q_eng (int): Quality score for English.
        q_math (int): Quality score for Mathematics.

    Returns:
        bool: True if all scores are of type int False otherwise.
    """
    return all(isinstance(score, int) and 0 <= score <= 100 for score in [q_kiny, q_eng, q_math])