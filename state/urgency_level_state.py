"""
Urgency Level for P3 Students in RL Environment

This module figures out how urgently a school needs new textbooks for P3 students (Kinyarwanda, English, Mathematics). No specific urgency data exists, so it’s simulated based on shortages.

What It Does:
- Shows if a school needs textbooks urgently (High), somewhat (Medium), or not much (Low).
- Ideal: All schools have enough textbooks (1:1 ratio, good quality).
- Reality: Shortages (1:7 to 1:223) and poor-quality books mean some schools need new books urgently.

How It’s Picked:
- Urgency is rated as Low (minor shortages), Medium (moderate shortages), or High (severe shortages).
- Most schools (50%) have Medium urgency (e.g., 1:7 to 1:50 ratios), 30% have High (e.g., 1:50 to 1:223, poor quality), and 20% have Low (e.g., close to 1:1).
- No difference between rural (70%, ~89 students) and urban (30%, ~156 students) schools, as shortages occur in all provinces.
- Example: A school with High urgency might have very few books (e.g., 1:100) or worn-out books.

Check for Realism:
- Urgency is checked to be Low, Medium, or High.
- Number of students (S) is checked to be in the valid range (30–200).

Purpose:
- Generates an urgency level for each school, showing how badly they need new textbooks.
- The RL agent uses (S, Urgency_Level) to prioritize schools with High urgency for faster deliveries.
"""

import numpy as np

def sample_urgency_level():
    """
    Generates the urgency level for a school needing new textbooks.

    Returns:
        int: Urgency level (0: Low, 1: Medium, 2: High).
    """

    # Sample urgency category based on probabilities
    categories = ['Low', 'Medium', 'High']
    probabilities = [0.2, 0.5, 0.3]  # 20% Low, 50% Medium, 30% High
    urgency_str = np.random.choice(categories, p=probabilities)

    # Map string to integer
    urgency_map = {'Low': 0, 'Medium': 1, 'High': 2}
    urgency_int = urgency_map[urgency_str]

    return urgency_int

def verify_urgency_level(urgency):
    """
    Verifies that the urgency level is within the valid range.

    Args:
        urgency (str): Urgency level ('Low', 'Medium', 'High').

    Returns:
        bool: True if urgency is 'Low', 'Medium', or 'High', False otherwise.
    """
    return urgency in ['Low', 'Medium', 'High']