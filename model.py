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

# removed label encoders due to the categorical-to-binary simplification

"""
Removed categorical encoders as they are no longer needed on making previously categorical vals binary.
"""

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

## we add a feed forward function that is self defined and passes an input
    ## parameter x through the network, returning its output from inputting x.

    def forward(self, x):
        return self.network(x)

# input_dim is 9 since there are 9 training features.
# the [1] index refers to the columns instead of the rows. i.e.,
# using [0] would call all the rows instead.

input_dim = X_train_tensor.shape[1]

# instantiating the model in the name 'model', running out input_dim (9 features) through it

model = RiskModel(input_dim)

# Define loss and optimizer
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

"""
setting the epochs as 200 reduces training error from  8.0753 to 0.1921
which is obviously a reasonable amount or error (well below 1.)

It seems like a good balance between time to run, and error.
"""
# Train the model
epochs = 200

"""
For each epoch, we:
"""

for epoch in range(epochs):
    # sets the model to training mode, which is apparently a good practice regardless of anything:
    model.train()
    # zeroing out previously created gradients before the next backpropagation through the layers
    # preventing gradient accumulation:
    optimizer.zero_grad()
    # runs data through the model to get predicted risk scores. these will be compared to the true scores to see error:
    output = model(X_train_tensor)
    # comparing true values to the model's outputs:
    loss = criterion(output, y_train_tensor)
    # inbuilt backward pass function that is applied to the loss
    # essentially turning around and asking "where did we go wrong?" for each epoch:
    loss.backward()
    # instructing the model to update weights using the gradients to reduce loss - the point of backpropagation
    # essentially the model's way of 'implementing feedback':
    optimizer.step()

    # Optional: set the model to eval mode - "stop learning and start testing your 'learning'"
    model.eval()
    # I think of torch.no_grad() ass us telling the model to "stop learning - just observe".
    with torch.no_grad():
        # while the gradients are not being changed, (learning mode is off),
        # we compare the model's learned risk with the true y values (y_test_tensor)
        test_output = model(X_test_tensor)
        test_loss = criterion(test_output, y_test_tensor)


    print(f"Epoch {epoch + 1}/{epochs} | Train Loss: {loss.item():.4f} | Test Loss: {test_loss.item():.4f}")



# Optionally: Save model
torch.save(model.state_dict(), "risk_model.pth")


"""
TEST ON NEW DATA:
"""


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

# converting it into a dataframe as models run on dataframes:

test_data_df = pd.DataFrame(test_data)

# encoders for categorical data

print(df.columns.tolist())


scaler = StandardScaler()
test_data_scaled = scaler.fit_transform(test_data_df)

# Loading Model & Weights
input_dim = test_data_scaled.shape[1]
# our earlier created RiskModel()
model = RiskModel(input_dim)
# Since we saved the model as torch.save(model.state_dict(), "risk_model.pth"),
model.load_state_dict(torch.load("risk_model.pth"))
#set the model into evaluation ("use what you've learnt") mode.
model.eval()

"""
finally we convert the scaled applicant data to torch's preferred torch tensor format,
which uses float32 with two extra zeros for the model. for reference, lets print the:

- test_tensor
- test_data_scaled

to see the difference.

"""
print('Test_data_scaled:\n\n', test_data_scaled)
test_tensor = torch.tensor(test_data_scaled, dtype=torch.float32)
print('test_tensor:\n\n',test_tensor)
