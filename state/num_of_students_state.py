"""
P3 Student State Definition for RL Textbook Distribution Environment

This module defines the state feature for the number of P3 students per school in the RL environment
for textbook distribution in Rwanda, focused on P3 students. The calculations use 2021/2022 Rwanda
educational statistics and account for school status (public, government-subsidized, private), dropout,
repetition rates, and a rural/urban split.

Data Points:
- Total P3 students enrolled: 416,259
- Number of primary schools: 3,831
  - Public: 1,316 schools
  - Government-subsidized: 1,897 schools
  - Private: 618 schools
- Average pupils per school (P1–P6):
  - Public: 834 students
  - Government-subsidized: 795 students
  - Private: 222 students
- Number of P3 classrooms: 6,519
- Average pupils per P3 classroom: 64
- P3 dropout rate: 6.7%
- P3 repetition rate: 23%

Calculation Steps:
1. Total P3 Students:
   - Data: 416,259 P3 students across 3,831 schools (2021/2022 statistics).
   - Average: 416,259 ÷ 3,831 = 108.66 ≈ 109 P3 students per school.
2. Average P3 Students by School Status (Unadjusted, P1–P6 ÷ 6 grades):
   - Public (1,316 schools): 834 students ÷ 6 ≈ 139 P3 students.
   - Government-subsidized (1,897 schools): 795 students ÷ 6 ≈ 133 P3 students.
   - Private (618 schools): 222 students ÷ 6 ≈ 37 P3 students.
   - Weighted average: (1,316 × 139 + 1,897 × 133 + 618 × 37) ÷ 3,831 = 458,091 ÷ 3,831 ≈ 119.6 (overestimate due to dropout/repetition).
3. Adjust for Dropout (6.7%) and Repetition (23%):
   - Effective students = Nominal × (1 + 0.23) × (1 - 0.067) = Nominal × 1.14759.
   - Total: 416,259 = 3,831 × (Nominal × 1.14759) → Nominal ≈ 94.67.
   - Effective per school: 94.67 × 1.14759 ≈ 108.64 ≈ 109.
   - Adjusted by status:
     - Public: 139 ÷ 1.14759 ≈ 121 P3 students.
     - Government-subsidized: 133 ÷ 1.14759 ≈ 116 P3 students.
     - Private: 37 ÷ 1.14759 ≈ 32 P3 students.
4. Classroom Validation:
   - Classrooms: 6,519 ÷ 3,831 ≈ 1.70 ≈ 1–2 P3 classrooms per school.
   - Students per classroom: 416,259 ÷ 6,519 ≈ 64 (matches data).
   - Public: 121 ÷ 64 ≈ 1.89 classrooms (~2).
   - Government-subsidized: 116 ÷ 64 ≈ 1.81 classrooms (~2).
   - Private: 32 ÷ 64 ≈ 0.5 classrooms (~1).
5. Range Estimation:
   - Minimum: 30 students (small private/rural school, 1 classroom of 30–50 students).
   - Maximum: 200 students (large public school, 2–3 classrooms × 64–100 students).
6. Rural/Urban Split for P3 Students Distribution:
    - Split: 70% rural, 30% urban, based on Rwanda’s demographic trend (~70% rural population, per World Bank data), as no specific rural/urban school data was provided.
    - Rural Mean (89): Chosen to reflect smaller schools (e.g., private: 32 P3 students, small public: ~121), as rural schools typically have fewer students (1–2 classrooms, ~64–128 students).
    - Urban Mean (156): Chosen for larger schools (e.g., public: 121, government-subsidized: 116, up to 2–3 classrooms, ~128–192 students), as urban schools have higher enrollment.
    - Weighted Mean: (0.7 × 89) + (0.3 × 156) = 62.3 + 46.8 ≈ 109, matching the overall average (416,259 ÷ 3,831).
    - Rural SD (20): Reflects lower variability in smaller rural schools (range 49–129, adjusted to 40–130).
    - Urban SD (30): Reflects higher variability in larger urban schools (range 96–216).
    - Weighted SD: sqrt(0.7 × 20^2 + 0.3 × 30^2) = sqrt(280 + 270) ≈ 23.45 ≈ 24, giving a range of 61–157 (±2 SD), extended to 30–200 for edge cases.
    - Purpose: Models realistic school size variability (small rural to large urban) while ensuring the average matches the data.
7. Final Range and Distribution:
   - Range: 30–200 P3 students per school.
   - Distribution: Mixture of normals (70% rural: mean = 89, SD = 20; 30% urban: mean = 156, SD = 30).
   - Clipped to 30–200 to avoid outliers.
"""

import numpy as np

class numStudentState:
    def __init__(self):
        self.rural_probability = 0.7
        self.rural_mean = 100
        self.rural_sd = 30
        self.urban_mean = 150
        self.urban_sd = 40
        self.num_students_min = 30
        self.num_students_max = 1000  # Updated from 200 to 1000 to utilize the flexibility in my system

    def sample_num_students(self):
        """
        Sample the number of P3 students for a school from a mixture of normal distributions
        (70% rural, 30% urban), clipped to the defined range.

        Returns:
            int: Number of P3 students, clipped between 30 and 1000.
        """
        if np.random.random() < self.rural_probability:
            num_students = np.random.normal(self.rural_mean, self.rural_sd)
        else:
            num_students = np.random.normal(self.urban_mean, self.urban_sd)
        return int(np.clip(num_students, self.num_students_min, self.num_students_max))

    def verify(self, num_students):
        """
        Verify if the number of students is within the acceptable range.

        Args:
            num_students (int): Number of students to verify.

        Returns:
            bool: True if valid, False otherwise.
        """
        return isinstance(num_students, (int, np.integer)) and self.num_students_min <= num_students <= self.num_students_max