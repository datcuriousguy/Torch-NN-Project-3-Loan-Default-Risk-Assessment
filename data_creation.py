
"""
Code to generate training data for training the model to identify low risk loan applicants
"""

import pandas as pd
import numpy as np
import random

# Set seed for reproducibility
np.random.seed(42)

# Employment status categories
employment_statuses = ["Stable Job", "Unstable Job", "Unemployed", "Recently Hired", "Student"]
employment_weights = [0.45, 0.2, 0.1, 0.15, 0.1]  # trying to use realistic ratios. Large proportion of recent hires and stable jobs

# Loan purpose categories
loan_purposes = ["Productive and/or Secure", "Non-productive and/or Less Secure"]
purpose_weights = [0.6, 0.4]

"""
our column headers
columns = [
    "credit_score",
    "dti_ratio",
    "annual_income",
    "loan_amount",
    "employment_status",
    "past_defaults",
    "credit_inquiries_12m",
    "loan_term",
    "loan_purpose",
    "loan_default"
]
"""

# Generator function for one row
def generate_row():

    """
    to try and mimic real world credit scores, we center the avg credit score around 680.
    (gaussian curve centered at 650)

    
    """
