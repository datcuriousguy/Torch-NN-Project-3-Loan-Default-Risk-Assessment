# Torch-NN-Project-3-Loan-Default-Risk-Assessment
This project focuses on automating loan default prediction in a banking environment using AI. We developed a neural network with PyTorch that analyzes applicant data to estimate the probability of default. The model outputs either a probability score or a binary decision to support efficient and scalable loan approval processes.

---

## The Situation

We are working in a bank or financial institution that offers customers the opportunity to take a loan. As most banks do, we need to understand which loan applicants are more and less likely to return the money with interest. That is, who is more likely to default.

## The Problem

Due to the large number of loan applicants, it can be costly and time-consuming to manually go through every application. Hence, automating the process with an AI Model would be helpful.

## Dealing with The Problem

We design a neural network using PyTorch / Torch that takes information about the loan applicant as an input and returns a probability of defaulting or a rounded binary value indicating (in a yes/no format) if they are likely to default or not.

Files:

- **`Readme.md`**
    
    **This file.**
    
- `sample_data.md`
    
    A sample of training data to be used to train the model.
    
- **`model.py`**
    
    The python code for the neural network alone, including hardcoded hyperparameters.
    
- **`resources.md`**
    
    A text file containing some resources on data for training and the significance of each of the features used to train the model.
    
- `database_control.py`
    
    This is a python file whose sole purpose is to create and then make manipulations to the MySQL database which will store our training data - which is also created in this file.
    
- `requirements.md`
    
    Lists the libraries required to run the neural network and get a desired output.
    
- Database information (Local MySQL Database):
    - Username: `daniel`
    - Password: `daniel`
    - Database name: `nn_project_2_database`

---

## Data Source

I have used `database_control.py` to: 

- Create the data to be used for training the neural network. I have made sure the parameters (see docstrings) are as realistic as possible.
- Load the data into a local MySQL Database (The purpose of doing it like this is to try and simulate how data is stored prior to training models in real world business scenarios)

---

## Understanding the Significance of the Data

Some features of the data for training that will be used by the model to understand (and then decide) the probability of defaulting on the loan are:

- Credit Score:
    
    ≥700 is considered good
    
- Debt-to-income Ratio:
    
    float value for which, lower the better
    
    - <0.35 is good
    - >0.5 indicates debt consists of signfiicant portion of their expenses
- Annual Income:
    
    float or int. Is it high enough to cover existing and potentially upcoming loans, as well as their existing expenses?
    
- Loan Amount:
    
    This is the amount the customer would like to borrow. A low risk applicant would have a low amount relative to their income, and a high risk one would have a high amount relative to their income.
    
- Employment status
    
    This data is categorical, and can be transformed later in the model code. Essentially we have around 5 categories:
    
    - Stable Job
    - Unstable Job
    - Unemployed
    - Recently Hired
    - Student
 
- Past Defaults
    
    Has the applicant defaulted (i.e., failed to pay back) any loans in the past? As a rule of thumb, the lower this number (0 or more, int) the better.
---

## How The Neural Network is Trained

1. Forward Pass to get predictions
2. Compute Loss of those predictions
3. Backpropagage to compute gradients 
    
    (Find out how much the error changes for a given change in gradient
    
4. Recompute weights 
    
    (New weight = Old weight – Learning rate × Gradient)
    
5. Repeat for all batches, if any

---

## Best Practices while Creating The Neural Network (From other GitHub Projects)

1. Using a Modular Architecture by making use of layers, modular functions, feed-forward and feed-backward functions, loss-functions, and optimizers. This is good for readability and makes things reusable as well.
2. Defining and implementing ways to handle missing values, normalizing features (if needed), and taking note of outliers are all recommended practices in the GitHub Community. Data cleaning and preprocessing prior to use for training will make sure that the neural network is properly trained while avoiding or reducing the chance for anomalous results.
3. Using regularization (minimizing needless complexity and exposing the network to more diverse data) will help prevent overfitting while improving generalization, which is desired in most neural networks. Note: There are multiple methods for regularization.
4. Optional: Tuning Hyperparameters like epochs and learning rate specific to strategies such as grid search or Beayesian optimization, can enhance model performance.
5. Taking note of the Bias and Variance of a given model is important as it enables the tuning of hyperparameters to match different scenarios (Essentially finding a point between Linear Regression and overfitting).
6. Obviously, a well-written README file (hello there) and good directory structure are useful to have for any given project.

---

## Sample Output
