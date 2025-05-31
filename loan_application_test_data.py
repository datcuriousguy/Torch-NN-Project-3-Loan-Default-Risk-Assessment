
"""This is Loan Applicant data to run through the model and get an output"""

test_data = [
    {
        "credit_score": 720,
        "debt_to_income": 0.25,
        "average_salary": 85000,
        "loan_amount": 21250,
        "employment_duration": 8,
        "past_defaults": 0,
        "credit_inquiries": 2,
        "loan_term": 36,
        "loan_purpose": "1"
    },
    {
        "credit_score": 610,
        "debt_to_income": 0.45,
        "average_salary": 42000,
        "loan_amount": 18900,
        "employment_duration": 2,
        "past_defaults": 1,
        "credit_inquiries": 3,
        "loan_term": 48,
        "loan_purpose": "0"
    },
    {
        "credit_score": 680,
        "debt_to_income": 0.3,
        "average_salary": 60000,
        "loan_amount": 18000,
        "employment_duration": 6,
        "past_defaults": 0,
        "credit_inquiries": 1,
        "loan_term": 24,
        "loan_purpose": "1"
    },
    {
        "credit_score": 590,
        "debt_to_income": 0.5,
        "average_salary": 35000,
        "loan_amount": 17500,
        "employment_duration": 1,
        "past_defaults": 2,
        "credit_inquiries": 4,
        "loan_term": 60,
        "loan_purpose": "0"
    },
    {
        "credit_score": 755,
        "debt_to_income": 0.2,
        "average_salary": 120000,
        "loan_amount": 24000,
        "employment_duration": 9,
        "past_defaults": 0,
        "credit_inquiries": 1,
        "loan_term": 36,
        "loan_purpose": "1"
    },
    {
        "credit_score": 645,
        "debt_to_income": 0.35,
        "average_salary": 56000,
        "loan_amount": 19600,
        "employment_duration": 0,
        "past_defaults": 0,
        "credit_inquiries": 2,
        "loan_term": 24,
        "loan_purpose": "1"
    },
    {
        "credit_score": 700,
        "debt_to_income": 0.3,
        "average_salary": 90000,
        "loan_amount": 27000,
        "employment_duration": 3,
        "past_defaults": 0,
        "credit_inquiries": 3,
        "loan_term": 36,
        "loan_purpose": "1"
    },
    {
        "credit_score": 550,
        "debt_to_income": 0.6,
        "average_salary": 30000,
        "loan_amount": 18000,
        "employment_duration": 0,
        "past_defaults": 3,
        "credit_inquiries": 5,
        "loan_term": 72,
        "loan_purpose": "0"
    },
    {
        "credit_score": 705,
        "debt_to_income": 0.4,
        "average_salary": 65000,
        "loan_amount": 26000,
        "employment_duration": 10,
        "past_defaults": 0,
        "credit_inquiries": 0,
        "loan_term": 24,
        "loan_purpose": "1"
    },
    {
        "credit_score": 600,
        "debt_to_income": 0.5,
        "average_salary": 40000,
        "loan_amount": 20000,
        "employment_duration": 2,
        "past_defaults": 2,
        "credit_inquiries": 4,
        "loan_term": 60,
        "loan_purpose": "0"
    },
    {
        "credit_score": 670,
        "debt_to_income": 0.2,
        "average_salary": 50000,
        "loan_amount": 10000,
        "employment_duration": 1,
        "past_defaults": 0,
        "credit_inquiries": 1,
        "loan_term": 12,
        "loan_purpose": "1"
    },
    {
        "credit_score": 740,
        "debt_to_income": 0.15,
        "average_salary": 110000,
        "loan_amount": 16500,
        "employment_duration": 7,
        "past_defaults": 0,
        "credit_inquiries": 1,
        "loan_term": 24,
        "loan_purpose": "1"
    },
    {
        "credit_score": 520,
        "debt_to_income": 0.7,
        "average_salary": 25000,
        "loan_amount": 17500,
        "employment_duration": 0,
        "past_defaults": 4,
        "credit_inquiries": 6,
        "loan_term": 72,
        "loan_purpose": "0"
    },
    {
        "credit_score": 690,
        "debt_to_income": 0.4,
        "average_salary": 72000,
        "loan_amount": 28800,
        "employment_duration": 4,
        "past_defaults": 1,
        "credit_inquiries": 2,
        "loan_term": 36,
        "loan_purpose": "1"
    },
    {
        "credit_score": 630,
        "debt_to_income": 0.3,
        "average_salary": 47000,
        "loan_amount": 14100,
        "employment_duration": 2,
        "past_defaults": 1,
        "credit_inquiries": 3,
        "loan_term": 12,
        "loan_purpose": "1"
    },
    {
        "credit_score": 780,
        "debt_to_income": 0.2,
        "average_salary": 135000,
        "loan_amount": 27000,
        "employment_duration": 10,
        "past_defaults": 0,
        "credit_inquiries": 1,
        "loan_term": 36,
        "loan_purpose": "1"
    },
    {
        "credit_score": 585,
        "debt_to_income": 0.55,
        "average_salary": 39000,
        "loan_amount": 21450,
        "employment_duration": 1,
        "past_defaults": 2,
        "credit_inquiries": 5,
        "loan_term": 60,
        "loan_purpose": "0"
    },
    {
        "credit_score": 710,
        "debt_to_income": 0.25,
        "average_salary": 95000,
        "loan_amount": 23750,
        "employment_duration": 6,
        "past_defaults": 0,
        "credit_inquiries": 2,
        "loan_term": 24,
        "loan_purpose": "1"
    },
    {
        "credit_score": 640,
        "debt_to_income": 0.4,
        "average_salary": 58000,
        "loan_amount": 23200,
        "employment_duration": 3,
        "past_defaults": 1,
        "credit_inquiries": 3,
        "loan_term": 48,
        "loan_purpose": "0"
    },
    {
        "credit_score": 690,
        "debt_to_income": 0.3,
        "average_salary": 67000,
        "loan_amount": 20100,
        "employment_duration": 0,
        "past_defaults": 0,
        "credit_inquiries": 2,
        "loan_term": 24,
        "loan_purpose": "1"
    }
]
