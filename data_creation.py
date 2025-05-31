
"""
Code to generate training data for training the model to identify low risk loan applicants
"""

# Note: On the categorical-to-binary simplification, I updated the database with the new data

import pandas as pd
import numpy as np
import random
import pymysql


# Database connection setup
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='password',
    database='nn_project_3_database',
    charset='utf8mb4',
    cursorclass=pymysql.cursors.DictCursor
)

# Set seed 42 (arbitrary) for reproducibility.
# Now data will be the same yet still random in future runs.
np.random.seed(42)

# Employment status categories
employment_durations = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
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
    "employment_duration",
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

    5. EMPLOYMENT DURATION (PREVIOUSLY EMPLOYMENT STATUS)
    Changed from 'student, newly employed, etc.' to '0,1,2,3,4' int duration of work. A duration of 0 indicates
    a new employee and a duration of 10 would indicate high job stability.

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

    9. LOAN PURPOSE (previously string, now binary int for simplicity.)
    random choice between 1 and 0.
    1 indicates productive asset purchase / use case (say, education)
    0 indicates non-productive asset / use case (say, buying a TV)
    """

    credit_score = int(np.clip(np.random.normal(680, 50), 300, 850))
    dti_ratio = round(np.clip(np.random.normal(0.35, 0.1), 0.1, 0.9), 2)
    annual_income = int(np.clip(np.random.normal(60000, 20000), 10000, 200000))
    loan_amount = int(np.clip(np.random.normal(annual_income * dti_ratio, 5000), 1000, 75000))
    employment_duration = random.choice(employment_durations)
    past_defaults = np.random.choice([0, 1], p=[0.85, 0.15]) if credit_score > 650 else np.random.choice([0, 1], p=[0.6, 0.4])
    credit_inquiries = int(np.clip(np.random.poisson(2), 0, 10))
    loan_term = int(np.random.choice([12, 24, 36, 48, 60, 72], p=[0.1, 0.2, 0.3, 0.2, 0.15, 0.05]))
    loan_purpose = random.choice([1,0])

    """
    The risk score calculation essentially works such that, the
     risk score increases proportionately with the factors:
     
    - dti_ratio (higher dti ratios attract higher weight as * 1.5)
    - past_defaults (If the person has defaulted before, this adds 2 to their risk)
    - credit_inquiries (Many inquiries = desperate borrower = potential risk)
    
    if loan_purpose == 0 O (indicating a non-productive asset purchase), we increase risk score by 1.
    And if the loan term is longer than 48 years, we multiply that number by 0.5 and add
    it to the risk score.
    """

    # Risk score calculation
    risk_score = (
        (700 - credit_score) * 0.01 +
        dti_ratio * 1.5 +
        past_defaults * 2 +
        credit_inquiries * 0.2 +
        (employment_duration < 1) * 1 +
        (loan_purpose == 0) * 1 +
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
        employment_duration,
        past_defaults,
        credit_inquiries,
        loan_term,
        loan_purpose,
        round(risk_score, 2)
    )

# SQL table creation
create_table_query = """
CREATE TABLE IF NOT EXISTS loan_training_data (
    id INT AUTO_INCREMENT PRIMARY KEY,
    credit_score INT,
    dti_ratio FLOAT,
    annual_income INT,
    loan_amount INT,
    employment_duration INT,
    past_defaults TINYINT,
    credit_inquiries INT,
    loan_term INT,
    loan_purpose INT,
    risk_score FLOAT
);
"""

# Column headers
print("credit_score | dti_ratio | annual_income | loan_amount | employment_duration     | past_defaults | credit_inquiries | loan_term | loan_purpose                    | risk_score")
print("-" * 130)

# Insert generated rows into MySQL
try:
    with connection.cursor() as cursor:
        cursor.execute(create_table_query)
        for _ in range(7000):
            row = generate_row()

            # Print row in tabular format
            print(f"{row[0]:<10} | {row[1]:<7} | {row[2]:<10} | {row[3]:<9} | {row[4]:<15} | {row[5]:<10} | {row[6]:<12} | {row[7]:<7} | {row[8]:<20} | {row[9]:<8}")

            insert_query = """
            INSERT INTO loan_training_data (
                credit_score, dti_ratio, annual_income, loan_amount,
                employment_duration, past_defaults, credit_inquiries,
                loan_term, loan_purpose, risk_score
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            """
            cursor.execute(insert_query, row)

        connection.commit()  # ensure the changes are made then permanently stored.

finally:
    connection.close()
