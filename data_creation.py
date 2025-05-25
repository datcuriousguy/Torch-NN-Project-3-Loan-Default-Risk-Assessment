
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
    1. CREDIT SCORE:
    to try and mimic real world credit scores, we center the avg credit score around 680.
    (gaussian curve centered at 650)
    clip ensures that the value stats similar to real credit scores. That is, ranging from 300 to 850

    2. DEBT-TO-INCOME:
    dti or debt to income ratio, is between clipped between 0.1 and 0.9 to avoid nonsensical values (like negative or greater than 100%)

    3. AVERAGE SALARY
    The average salary is clipped to stay realistic: minimum $10,000, maximum $200,000.

    4. LOAN AMOUNT (WHAT IS BEING BORROWED)
    for the loan amount, annual_income * dti_ratio estimates how much the person can afford to borrow.

    5. EMPLOYMENT STATUS
    for the employment status, we have a list ["Stable Job", "Unstable Job", "Unemployed", "Recently Hired", "Student"].
    out of these items, we pick any one at random but with some bias in the weights, namely a bias towards
    student and stable job, to try to match the constantly changing reality of the job market.

    6. NUMBER OF PAST DEFAULTS
    Past-default number indicates whether and if so, how many times the borrower has defaulted (or failed to pay back a loan) in
    the past.It works such that if their credit score is good then they are less likely to have defaulted.
    i.e., it is dependent on their credit score (input feature).

    7. NUMBER OF CREDIT INQUIRIES
    This is the number of times the applicant has applied for credit in the last year.
    - It can range between 0 and 10 and is simulated using a poisson distribution.

    8. LOAN TERM
    A set number of months between 12 and 72 mos (or 1 to 6 years).
    The probabilities [0.1, 0.2, 0.3, 0.2, 0.15, 0.05] are such that longer loans are rarer.

    9. LOAN PURPOSE
    60% Chance of productive / secure loan use (say, a car or basic two-wheeler)
    40% Chance of non-productive / insecure loan use (say, a TV)
    """

    credit_score = int(np.clip(np.random.normal(680, 50), 300, 850))
    dti_ratio = round(np.clip(np.random.normal(0.35, 0.1), 0.1, 0.9), 2)
    annual_income = int(np.clip(np.random.normal(60000, 20000), 10000, 200000))
    loan_amount = int(np.clip(np.random.normal(annual_income * dti_ratio, 5000), 1000, 75000))
    employment_status = random.choices(employment_statuses, weights=employment_weights, k=1)[0]
    past_defaults = np.random.choice([0, 1], p=[0.85, 0.15]) if credit_score > 650 else np.random.choice([0, 1], p=[0.6, 0.4])
    credit_inquiries = int(np.clip(np.random.poisson(2), 0, 10))
    loan_term = int(np.random.choice([12, 24, 36, 48, 60, 72], p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05]))
    loan_purpose = random.choices(loan_purposes, weights=purpose_weights, k=1)[0]

    """
    The risk score calculation essentially works such that, the
     risk score increases proportionately with the factors:
     
    - dti_ratio (higher dti ratios attract higher weight as * 1.5)
    - past_defaults (If the person has defaulted before, this adds 2 to their risk)
    - credit_inquiries (Many inquiries = desperate borrower = potential risk)
    
    if loan_purpose == "Non-productive and/or Less Secure", we increase risk score by 1.
    And if the loan term is longer than 48 years, we multiply that number by 0.5 and add
    it to the risk score.
    """

    # Risk score calculation
    risk_score = (
        (700 - credit_score) * 0.01 +
        dti_ratio * 1.5 +
        past_defaults * 2 +
        credit_inquiries * 0.2 +
        (employment_status != "Stable Job") * 1 +
        (loan_purpose == "Non-productive and/or Less Secure") * 1 +
        (loan_term > 48) * 0.5
    )

    # We decide the applicant is more likely to default on the loan if the risk score is greater than 4,
    # else loan default = 0 and they are safe. Remember this is just the training data.
    loan_default = 1 if risk_score > 4 else 0

    return (
        credit_score,
        dti_ratio,
        annual_income,
        loan_amount,
        employment_status,
        past_defaults,
        credit_inquiries,
        loan_term,
        loan_purpose,
        round(risk_score, 2)
    )
