"""
Location Type for P3 Students in RL Environment

This module figures out if a school is rural or urban for P3 students.

What It Does:
- Identifies the school’s location as Rural or Urban.
- Ideal: All schools have equal access to resources regardless of location.
- Reality: Rural schools (70%, ~89 students) face more challenges (e.g., poor infrastructure, fewer textbooks) than urban schools (30%, ~156 students).

How It’s Picked:
- Location is chosen randomly: 70% chance Rural, 30% Urban, based on census data.
- Example: A school might be tagged as Rural, indicating potential issues like poor storage or delivery delays.

Check for Realism:
- Location is checked to be Rural or Urban.
- Number of students (S) is checked to be in the valid range (30–200).

Purpose:
- Generates a location type for each school, showing its rural or urban status.
- The RL agent uses (S, Location_Type) to prioritize schools with location-specific challenges (e.g., rural schools needing more support).

Notes:
- Uses data from 3,831 primary schools, 6,519 P3 classrooms, 64 students/classroom.
- State has 2 parts: S, Location_Type.
- Rural (70%) and urban (30%) split per NISR 2022 Census.
- FLN Priority 3 and Auditor General’s reports (2021/2024) note rural challenges (e.g., shortages 1:7 to 1:223).
"""

import numpy as np

def sample_location_type():
    """
    Generates the location type for a school as an integer.

    Returns:
        int: Location type (0: Rural, 1: Urban).
    """
    # Sample location type based on probabilities using string mapping, then convert to integer
    location_str = np.random.choice(['Rural', 'Urban'], p=[0.7, 0.3])
    location_int = 0 if location_str == 'Rural' else 1
    return location_int

def verify_location_type(location):
    """
    Verifies and maps the location type to integer.

    Args:
        location (int or str): Location type (0, 1, 'Rural', 'Urban').

    Returns:
        int or bool: 0 for Rural, 1 for Urban, False if invalid.
    """
    if isinstance(location, str):
        if location == 'Rural':
            return True
        elif location == 'Urban':
            return True
        else:
            return False
    elif isinstance(location, int):
        if location == 0 or location == 1:
            return True
        else:
            return False
    else:
        return False