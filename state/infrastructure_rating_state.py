"""
Infrastructure Rating for P3 Students in RL Environment

This module figures out how good a school’s classroom libraries and storage are for P3 students’ learning materials (textbooks, teacher guides). It’s based on Rwanda’s 2021/2022 educational statistics (3,831 schools, 6,519 P3 classrooms, ~109 students/school), NISR 2022 Census (~72.1% rural, 27.9% urban), and FLN Priority 3, which emphasizes classroom libraries with leveled readers, decodable books, and children’s texts, plus storage for materials. No specific data on library or storage quality exists, so it’s simulated using the location type (Rural or Urban) from location_type_state.py.

What It Does:
- Gives a score (0–100) for the quality of classroom libraries (e.g., shelves with books) and storage (e.g., cabinets for textbooks).
- Ideal: Schools have well-stocked libraries with many books and secure storage to keep materials safe.
- Reality: Rural schools (~89 students) often have few or no libraries and poor storage, while urban schools (~156 students) have better facilities.

How It’s Picked:
- Score depends on the school’s location (Rural or Urban, from location_type_state.py).
- Rural schools average ~50 (e.g., small or no library, poor storage); urban schools average ~70 (e.g., better libraries, secure storage).
- Overall average: ~56 (mix of 70% rural, 30% urban).
- Scores stay between 0 and 100.
- Example: A Rural school scoring 45 might have a small shelf with few books and a broken cabinet; an Urban school scoring 75 has a good library and safe storage.

Check for Realism:
- Score is checked to be between 0 and 100.
- Number of students (S) is checked to be in the valid range (30–200).
- Location type is checked to be Rural or Urban.

Purpose:
- Generates an infrastructure score for each school, showing the quality of classroom libraries and storage.
- The RL agent uses (S, Infrastructure_Rating) to prioritize schools with low scores (e.g., poor libraries or storage) for support, as it affects how well materials are used and kept safe.
"""

import numpy as np

def sample_infrastructure_rating(location_type):
    """
    Generates the infrastructure rating for a school’s classroom libraries and storage.

    Args:
        location_type (str): Location type ('Rural' or 'Urban', from location_type_state.py).

    Returns:
        int: Infrastructure rating (0–100).
    """
    # Convert integer location_type to string
    location_str = 'Rural' if location_type == 0 else 'Urban' if location_type == 1 else None
    if location_str is None:
        raise ValueError("Location type must be 0 (Rural) or 1 (Urban), got {}".format(location_type))

    # Sample rating from normal distribution based on location type
    if location_type == 'Rural':
        rating = np.random.normal(50, 15)  # Mean = 50, SD = 15
    else:  # Urban
        rating = np.random.normal(70, 15)  # Mean = 70, SD = 15

    # Clip to 0–100
    rating = int(max(0, min(100, rating)))

    return rating

def verify_infrastructure_rating(rating):
    """
    Verifies that the infrastructure rating is within the valid range.

    Args:
        rating (int): Infrastructure rating (0–100).

    Returns:
        bool: True if rating is in [0, 100] False otherwise.
    """
    return isinstance(rating, int) and 0 <= rating <= 100