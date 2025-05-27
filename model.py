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

"""
Here we split the input features vs thde target variable.
X is all columns minus the risk_score (as that is the target).
y is the risk_score column alone.
"""
X = df.drop(columns=['risk_score'])
y = df['risk_score']

# defining the std scaler variable
scaler = StandardScaler()
# Scaling features to have a mean of 0 and std_dev of 1 (by default)
# prevents outlier-values from dominating and keeps data normalized.
X = scaler.fit_transform(X)

# splitting data into an 80-20 train-test ratio for evaluation
#note: 0.2 refers to the 20% and the remaining 80% is for training.
# datatype resulting from this: numpy arrays
# the random variable is arbitrary. I like the number 77
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=77)

"""
Now we convert our training X, training Y, testing X and testing Y into
torch's required torch-tensor datatype so that we can use it to train a
torch model. We haven't defined the model yet but will soon.

Note: torch.tensor() is the built in torch func we are using
We use .view(-1, 1) for y train and y test because, since x is already
a 2D array as a result of StandardScaler, we now need to convert the y train and
y test into 2D from its 1D form. 

Consider a function like nn.MSELoss(). it cannot simply run on y_train_tensor
in the form [num_samples]. it needs to be of the form [num_samples, 1] (or 2D).

Hence we use .view(-1, 1) where it is reshaped accordingly.
"""
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).view(-1, 1)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).view(-1, 1)


# Define the neural network
"""
Mapping features to loan default is a task with a level of variance that
is not as high as say, image recognition but definitely not linear either.
This is something to think about when choosing the number of neurons per hidden layer.

In general, a good starting point appears to be with 64 neurons for the
input layer, cutting down by 2 for further abstraction of features as we
move through the network (to next hidden layer).

Too few layers would hinder 'learning' of anything outside of linear,
and 
Too many layers would lead to a higher risk of overfitting.

A typical practice seems to be 2 hidden layers, so thatsa what we'll go for.

As for the Linear Layers, linear layers use the typical line formula:

y = xW + bias. This is not very different from the formula of a straight line.

This is helpful because the weights (or bias) controls how much each input
feature influences the output, quite literally.

Without ReLU, the network would just learn linear functions and not have enough 
abstraction. From my research, ReLU seems better for smaller neural networks like
this one, unline Tanh or Sigmoid, which might run a higher risk of the
vanishing gradient problem.

Hence, we go from input -> 64, 64 -> h1 (32), h1 (32) -> output (1)
"""

class RiskModel(nn.Module):
    def __init__(self, input_dim):
        super(RiskModel, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

## we add a feec forward function that is self defined and passes an input
    ## parameter x through the network, returning its output from inputting x.

    def forward(self, x):
        return self.network(x)
