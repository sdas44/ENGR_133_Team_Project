
"""
Course Number: ENGR 13300
Semester: e.g. Fall 2025

Description:
    This program calculates the actual age in years and seconds from the days elapse since the last birthday

Assignment Information:
    Assignment:     tp3 team 3
    Team ID:        LC05, 05
    Author:         Samarth Das, das316@purdue.edu
    Date:           10/9/2025

Contributors:
    Edward Ojuolape, eojuolap@purdue.edu,
    Lwanda Muigo, muigl01@purdue.edu
    Benjamin Tianming Sun, sun1384@purdue.edu

    My contributor(s) helped me:
    [X] understand the assignment expectations without
        telling me how they will approach it.
    [X] understand different ways to think about a solution
        without helping me plan my solution.
    [X] think through the meaning of a specific error or
        bug present in my code without looking at my code.
    Note that if you helped somebody else with their code, you
    have to list that person as a contributor here as well.

Academic Integrity Statement:
    I have not used source code obtained from any unauthorized
    source, either modified or unmodified; nor have I provided
    another student access to my code.  The project I am
    submitting is my own original work.
"""
import math
import pandas as pd
import numpy as np
import pathlib
import matplotlib.pyplot as plt

def load_dataset(file_path, feature_cols, label_col, shuffle, seed=70):
 """
 Loads a dataset from a CSV file, separates features and labels,
 and optionally shuffles the data.
 """
 df = pd.read_csv(file_path)
 X = df[feature_cols].to_numpy()
 y = df[label_col].to_numpy()

 if shuffle:
     np.random.seed(seed)
     shuffled_indices = np.random.permutation(len(y))
     X = X[shuffled_indices]
     y = y[shuffled_indices]

 return X, y


def train_val_test_split(X, y, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
  X_train_end = int(len(X) * train_ratio)
  X_val_end = X_train_end + int(len(X) * val_ratio)
  X_train, X_val, X_test = X[:X_train_end], X[X_train_end:X_val_end], X[X_val_end:]
  y_train, y_val, y_test = y[:X_train_end], y[X_train_end:X_val_end], y[X_val_end:]
  
  return X_train, y_train, X_val, y_val, X_test, y_test

def sigmoid(z):
    z = np.clip(z, -500, 500)
    epsilon = 1e-15
    return 1 / (1 + np.exp(-z) + epsilon)
  
def predict_proba(X, w, b):
  z = np.dot(X, w) + b
  z = sigmoid(z)
  
  return z

def predict_labels(X, w, b, threshold = 0.5):
  probability = predict_proba(X, w, b)
  
  labels = (probability >= threshold).astype(int)
  
  return labels

def compute_loss_and_grads(X, y, w, b):
  predictions = predict_proba(X, w, b)
  
  loss = np.mean(-y * np.log(predictions + 1e-15) - (1 - y) * np.log(1 - predictions + 1e-15))
  
  predicted_probability = predict_proba(X, w, b)
  
  loss_change_respect_w = 1 / X.shape[0] * np.dot(X.T, (predicted_probability - y))
  loss_change_respect_b = 1 / X.shape[0] * np.sum(predicted_probability - y)
  
  alpha = 0.01
  
  w -= alpha * loss_change_respect_w
  b -= alpha * loss_change_respect_b
  
  return loss, w, b
  
def train_logistic_regression(X_train, y_train, X_val=None, y_val=None,
                            learning_rate=0.01, num_iterations=1000):
    """
    Train logistic regression model using gradient descent

    Parameters:
    - X_train, y_train: training data
    - X_val, y_val: validation data (optional)
    - learning_rate: step size for gradient descent
    - num_iterations: number of training iterations

    Returns:
    - w_final, b_final: trained parameters
    - loss_history: list of loss values during training
    """

    # Get dimensions
    m, n = X_train.shape

    # Initialize parameters
    w = np.random.randn(n) * 0.01
    b = 0.0

    # Store loss history
    loss_history = []
    val_loss_history = []

    # Training loop
    for i in range(num_iterations):
        # Forward pass: compute loss and gradients
        loss, dw, db = compute_loss_and_grads(X_train, y_train, w, b)   

        # Update parameters
        w = w - learning_rate * dw
        b = b - learning_rate * db

        # Store loss
        loss_history.append(loss)

        # Optional: print progress every 100 iterations
        if i % 100 == 0:
            print(f"Iteration {i}, Loss: {loss:.4f}")

    return w, b, loss_history

def calculate_metrics(predicted_labels, true_labels):
  accuracy = np.mean(predicted_labels == true_labels)
  error_rate = 1 - accuracy
  
  
  return accuracy, error_rate



def main():
  file_path = pathlib.Path(input("Enter the path to the feature dataset: "))
  feature_cols = ['hue_mean', 'hue_std', 'saturation_mean', 'saturation_std', 'value_mean', 'value_std', 'num_lines', 'has_circle']
  label_col = ['ClassId']
  shuffle = input("Shuffle the dataset? (yes/no): ").strip().lower() == "yes"
  seed = int(input("Enter a seed for loading the dataset: "))
  
  #load dataset and create train, val, test splits
  X, y = load_dataset(file_path, feature_cols, label_col, shuffle, seed)
  X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X, y)
  
  #data preprocessing
  # Calculate statistics from training data
  feature_means = np.mean(X_train, axis=0)
  feature_stds = np.std(X_train, axis=0)

  # Standardize training data
  X_train_std = (X_train - feature_means) / feature_stds

  # Apply same transformation to validation/test data
  X_val_std = (X_val - feature_means) / feature_stds
  X_test_std = (X_test - feature_means) / feature_stds 
  
  #train model
  w_final, b_final, loss_history = train_logistic_regression(X_train_std, y_train, X_val_std, y_val,
                                                            learning_rate=0.01, num_iterations=1000)
  #evaluate model
  y_test_pred = predict_labels(X_test_std, w_final, b_final)
  accuracy, error_rate = calculate_metrics(y_test_pred, y_test)
  print(f"Test Accuracy: {accuracy:.4f}")
  print(f"Test Error Rate: {error_rate:.4f}")
  
  #plot loss history
  plt.plot(loss_history)
  plt.xlabel('Iteration')
  plt.title("Training Progress")
  
  


if __name__ == "__main__":
  main()