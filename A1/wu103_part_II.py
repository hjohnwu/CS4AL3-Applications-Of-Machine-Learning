# Credit for some sample/template/pilot code taken from course + lecture content:
# Source: python polynomial_regression.py ('Swati Mishra', Sep 9, 2024)
# Purpose: This python code includes Polynomial Regression
# License: MIT License
#--------------------------------------------------
# Purpose: Assignment 1 - Regression (Part II)
# Author: John Wu
# Student Number: WU103
# Created: Sep 10, 2024
#--------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Import data
data = pd.read_csv("training_data.csv")
# After plotting a pre-visual of the raw data from the CSV
# I noticed Height, and Viscera Weight had select extreme values
# Lets do some data cleaning and remove those to improve our model when training
# (Note) pre-visual plotting indicates to me that "polynomial relationship" best fits data
rows_to_drop = data[(data['Height'] > 0.5) | (data['Viscera_weight'] > 0.7)].index
data_cleaned = data.drop(rows_to_drop)

class polynomial_regression():
 
    def __init__(self, x_:list, y_:list) -> None:

        self.input = np.array(x_)
        self.target = np.array(y_)
        self.feature_mean = None
        self.feature_std = None
        self.target_mean = None
        self.target_std = None

    def preprocess(self, X_train=None, Y_train=None):
        # Use the full input and target for preprocessing by default
        # if training data is present, then specify input segments "for cross-validation purposes"
        if X_train is None:
            X_train = self.input
        if Y_train is None:
            Y_train = self.target

        # Normalize the input features
        hmean = np.mean(X_train, axis=0)
        hstd = np.std(X_train, axis=0)
        x_train_normal = (X_train - hmean) / hstd
        self.feature_mean = hmean
        self.feature_std = hstd

        # Generate polynomial features with degree n = 3
        X = np.column_stack([x_train_normal, x_train_normal ** 2, x_train_normal ** 3])

        # Normalize the target values
        gmean = np.mean(Y_train)
        gstd = np.std(Y_train)
        y_train_normal = (Y_train - gmean) / gstd
        self.target_mean = gmean
        self.target_std = gstd

        # Arrange in matrix format
        Y = np.array([y_train_normal]).T

        return X, Y

    def train(self, X, Y):
        # Compute and return beta using OLS
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    def predict(self, x_test, beta):
        # Normalize the test input identically to what we did for training data
        x_test_normal = (x_test - self.feature_mean) / self.feature_std
        # Generate polynomial features of degree n = 3, for test set
        X = np.column_stack([x_test_normal, x_test_normal ** 2, x_test_normal ** 3])
        # Predict y points using x points
        y_pred_normal = X.dot(beta)
        # Denormalize the y values to compare against non-normalized validation set
        y_pred = y_pred_normal * self.target_std + self.target_mean
        # Return the predicted y points
        return y_pred

    def MSE(self, Y, Y_hat):
        # Mean Square Error Cost Function
        mse = np.mean((Y - Y_hat) ** 2)
        return mse
    
    def k_fold_cross_validation(self, k):
        # Set fold lengths based on data size
        n = len(self.target)
        fold_size = n // k
        
        # Create an array of indices to fold over
        indices = np.arange(n)  
        best_mse = float('inf')
        best_beta = 0
        train_mse_values = []
        test_mse_values = []
        for i in range(k):
            # Get the indices for the test set
            test_indices = indices[i * fold_size: (i + 1) * fold_size]
            
            # Get the indices for the training set excluding the folds
            # train_indices are all indices except the test_indices
            train_indices = np.concatenate([indices[:i * fold_size], indices[(i + 1) * fold_size:]])
            
            # Split the data into training and testing sets
            X_train, X_test = self.input[train_indices], self.input[test_indices]
            Y_train, Y_test = self.target[train_indices], self.target[test_indices]
            
            # Preprocess the training and test data
            X, Y = self.preprocess(X_train, Y_train)
            beta = self.train(X, Y)
            
            # Predict on the training data - for train MSE calculation
            Y_train_pred = self.predict(X_train, beta)
            train_mse = self.MSE(Y_train.ravel(), Y_train_pred.ravel())
            train_mse_values.append(train_mse)
            
            # Predict on the test data - for test MSE calculation
            Y_test_pred = self.predict(X_test, beta)
            test_mse = self.MSE(Y_test.ravel(), Y_test_pred.ravel())
            test_mse_values.append(test_mse)
            
            # Finding best beta and MSE
            if test_mse < best_mse:
                best_mse = test_mse
                best_beta = beta
            
            # Calculating MSE averages to display and report    
            avg_train_mse = np.mean(train_mse_values)
            avg_test_mse = np.mean(test_mse_values)
        
        # Return the best beta and best MSE and print the average MSE across all folds
        print(f"\nBest Test MSE over {k} folds: {best_mse}")
        print(f"\nAverage Train MSE over {k} folds: {avg_train_mse}")
        print(f"\nAverage Test MSE over {k} folds: {avg_test_mse}")
        print(f'\nBest found corresponding "beta" values =\n {best_beta}')
        return best_beta


# Axis labels
features_list = ['Length', 'Diameter', 'Height', 'Whole_weight', 'Shucked_weight', 'Viscera_weight', 'Shell_weight']
target = 'Rings'

# Extract features from data
X = data_cleaned[features_list].values
Y = data_cleaned[target].values

# Instantiate Polynomial_Regression class    
pr = polynomial_regression(X, Y)

# Running K-Fold Cross Validation with k = 6 folds
best_beta = pr.k_fold_cross_validation(6)

# Preprocessing and normalizing data
X_, Y_ = pr.preprocess()

# Use the computed beta for prediction
Y_pred = pr.predict(X, best_beta)

# Removed a tiny handful of insignificant negative Ring/Age values for graph scale coherence
# You can see what I mean and how they look plotted by removing the abs() line of code below
Y_pred = np.absolute(Y_pred)

# Plotting all our data on 7 graphs for 7 features and veiwing them on 1 comprehensive figure
fig, axs = plt.subplots(3, 3, layout="constrained", figsize=(12, 8))
for i, ax in enumerate(axs.flat[:7]):
    ax.scatter(X[:, i], Y, color='b', label='Actual Values')
    ax.scatter(X[:, i], Y_pred, color='r', alpha=0.25, label='Predicted Values')
    ax.set_xlabel(features_list[i])
    ax.set_ylabel(target + " (+1.5 is Years)")
    ax.set_title(f'{features_list[i]} vs Age')
    ax.legend()
for ax in axs.flat[7:]:
    ax.set_visible(False)
    
plt.show()