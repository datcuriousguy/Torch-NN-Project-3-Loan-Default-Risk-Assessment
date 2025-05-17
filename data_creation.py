
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
    clip ensures that the value stats similar to real credit scores. That is, ranging from 300 to 850

    dti or debt to income ratio, is between clipped between 0.1 and 0.9 to avoid nonsensical values (like negative or greater than 100%)

    The average salary is clipped to stay realistic: minimum $10,000, maximum $200,000.

    for the loan amount, annual_income * dti_ratio estimates how much the person can afford to borrow.

    for the employment status, we have a list ["Stable Job", "Unstable Job", "Unemployed", "Recently Hired", "Student"].
    out of these items, we pick any one at random but with some bias in the weights, namely a bias towards
    student and stable job, to try to match the constantly changing reality of the job market.
    
    """

    credit_score = int(np.clip(np.random.normal(680, 50), 300, 850))
    dti_ratio = round(np.clip(np.random.normal(0.35, 0.1), 0.1, 0.9), 2)
    annual_income = int(np.clip(np.random.normal(60000, 20000), 10000, 200000))
    loan_amount = int(np.clip(np.random.normal(annual_income * dti_ratio, 5000), 1000, 75000))
    employment_status = random.choices(employment_statuses, weights=employment_weights, k=1)[0]
