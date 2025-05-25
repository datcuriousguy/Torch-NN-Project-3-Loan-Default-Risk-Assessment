import pymysql
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# Connect to MySQL and load data
connection = pymysql.connect(
    host='localhost',
    user='root',
    password='password',
    database='nn_project_3_database'
)

query = "SELECT * FROM loan_training_data"

# fetches all records of training data and loads them into a pandas dataframe,
# a convenient way to organize them with unique identifiers for each row.
df = pd.read_sql(query, connection)
connection.close()

# We don't need the ID column for testing or training.
df = df.drop(columns=['id'])

# Encode categorical features that aren't numeric
categorical_cols = ['employment_status', 'loan_purpose']
label_encoders = {}

"""
For each categorical column, we create a label-encoder that takes
the textual category name and converts it into integers, saving it
incase its needed later. Without this we wouldn't be able to train
the model on data like loan_purpose.
"""

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le  # Save encoder for possible use later
