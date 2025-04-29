import math, random, os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import joblib


csv_path_dataset = os.path.join("datasets", "mnist_train.csv" )
csv_path_testset = os.path.join("datasets", "mnist_test.csv" )
dataset = pd.read_csv(csv_path_dataset)
testset = pd.read_csv(csv_path_testset)

y_train = dataset.iloc[:, 0]
X_train = dataset.iloc[:, 1:]

y_test = testset.iloc[:, 0]
X_test = testset.iloc[:, 1:]


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the MLPClassifier with best parameters
model_filename = 'mlp_model.joblib'

try:
    # Load the model from file if it exists
    mlp = joblib.load(model_filename)
    print("Model loaded from file.")
except FileNotFoundError:
    # If the file doesn't exist, create a new model
    mlp = MLPClassifier(
        hidden_layer_sizes=(500, 200),  # Two hidden layers with 500 and 200 neurons
        activation='relu',              # ReLU activation function
        solver='adam',                  # Adam optimizer
        alpha=0.0001,                   # L2 regularization parameter
        max_iter=1000,                  # Maximum number of iterations
        random_state=42                 # For reproducibility
    )

    # Train the model on the scaled training data
    mlp.fit(X_train_scaled, y_train)

    # Save the trained model to a file
    joblib.dump(mlp, model_filename)
    print("Model trained and saved to file.")

# Make predictions on the test set
y_pred = mlp.predict(X_test_scaled)

# Calculate and print the accuracy
#accuracy = accuracy_score(y_test, y_pred)
#print(f"Accuracy: {accuracy:.4f}") # Accuracy: 0.9788

# Create a Pandas DataFrame for the submission
submission = pd.DataFrame({'ID': range(1, len(X_test) + 1), 'Label': y_pred})

# Save the DataFrame to a CSV file
submission.to_csv('submission.csv', index=False)
