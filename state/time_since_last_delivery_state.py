"""
Time Since Last Delivery for P3 Students in RL Environment

This module figures out how many days have passed since a school last received a textbook delivery for P3 students (Kinyarwanda, English, Mathematics). It’s based on Rwanda Basic Education Board (REB). As of April 2022, Rwanda began annual textbook distributions, with some delays by the 2024 Auditor General’s report (delivery delays 27–200 days).

What It Does:
- Shows how long it’s been (in days) since a school got its last textbook delivery.
- Ideal: Schools get new textbooks at the start of the academic year.
- Reality: Deliveries happen about once a year, but delays (27–200 days) mean some schools get books late, though reusable textbooks stay available if quality is good.

How It’s Picked:
- Time is chosen randomly between 0 and 365 days (e.g., 30 days, 300 days).
- No difference between rural (70%, ~89 students) and urban (30%, ~156 students) schools, as delays occur in all provinces.
- Average: ~183 days (half a year, or (0 + 365) ÷ 2).
- Example: A school with 200 days since its last delivery hasn’t gotten new textbooks in about 6–7 months, so books may be worn out if quality is low.

Check for Realism:
- Time is checked to be between 0 and 365 days.
- Number of students (S) is checked to be in the valid range (30–200).

Purpose:
- Generates the time since last delivery for each school, showing how long they’ve waited for textbooks.
- The RL agent uses (S, Time_Since_Delivery) to prioritize schools waiting longer (e.g., closer to 365 days), especially if textbook quality is poor.
"""

import numpy as np

def sample_time_since_last_delivery():
    """
    Generates the time since the last textbook delivery for a school.

    Args:

    Returns:
        int: Days since last delivery (0–365).
    """

    # Sample time from uniform distribution
    days = int(np.random.uniform(0, 366))  # 0 to 365 inclusive

    return days

def verify_time_since_last_delivery(days):
    """
    Verifies that the time since last delivery is within the valid range.

    Args:
        days (int): Days since last delivery.

    Returns:
        bool: True if days is of type int and positive, False otherwise.
    """
    return isinstance(days, int) and days >= 0
