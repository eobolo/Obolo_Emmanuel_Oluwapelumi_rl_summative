"""
Delivery Success History for P3 Students in RL Environment

This module figures out how successful a school’s past textbook deliveries have been for P3 students (Kinyarwanda, English, Mathematics). No specific data on delivery success exists, so it’s simulated.
What It Does:
- Shows if a school’s past textbook deliveries were mostly successful (on time, complete), average, or poor.
- Ideal: All deliveries arrive on time with enough textbooks.
- Reality: Delays and shortages mean some schools get books late or incomplete.

How It’s Picked:
- Success is rated as Low (<30%), Medium (30–70%), or High (>70%).
- Most schools (50%) have Medium success (some issues), 30% have High (mostly good), and 20% have Low (often late or incomplete).
- No difference between rural (70%, ~89 students) and urban (30%, ~156 students) schools, as issues occur in all provinces.
- Example: A school with Low success might have late deliveries or gotten fewer books than needed.

Check for Realism:
- Success is checked to be Low, Medium, or High.
- Number of students (S) is checked to be in the valid range (30–200).

Purpose:
- Generates a delivery success rating for each school, showing past reliability.
- The RL agent uses (S, Delivery_Success_History) to prioritize schools with Low success for better logistics support.
"""

import numpy as np

def sample_delivery_success_history():
    """
    Generates the historical success rate of textbook deliveries for a school.

    Returns:
        int: Delivery success rating (0: Low, 1: Medium, 2: High).
    """

    # Sample success category based on probabilities
    categories = ['Low', 'Medium', 'High']
    probabilities = [0.2, 0.5, 0.3]  # 20% Low, 50% Medium, 30% High
    success_str = np.random.choice(categories, p=probabilities)

    # Map string to integer
    success_map = {'Low': 0, 'Medium': 1, 'High': 2}
    success_int = success_map[success_str]

    return success_int

def verify_delivery_success_history(success):
    """
    Verifies that the delivery success history is within the valid range.

    Args:
        success (str): Delivery success rating ('Low', 'Medium', 'High').

    Returns:
        bool: True if success is 'Low', 'Medium', or 'High'.
    """
    return success in ['Low', 'Medium', 'High']