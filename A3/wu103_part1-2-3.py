# Credit for some sample/template/pilot code taken from course + lecture content:
# Source: python Assignment_3.py ('Swati Mishra', Sep 23, 2024)
# Purpose: This python includes assignment boiler plate code
# License: MIT License
#--------------------------------------------------
# Purpose: Assignment III
# Author: John Wu
# Student Number: WU103
# Created: Nov 10, 2024
#--------------------------------------------------

from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn.model_selection import train_test_split 
from sklearn.metrics import accuracy_score, recall_score, precision_score
from sklearn.utils import shuffle

class svm_():
    def __init__(self, learning_rate, epoch, C_value, X, Y):
        # Initialize variables
        self.input = X
        self.target = Y
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.C = C_value
        self.weights = np.zeros(X.shape[1])

    def compute_gradient(self, X, Y):
        # Organize the array as vector
        X_ = np.array([X])
        # Hinge loss gradient calculation
        hinge_distance = 1 - (Y * np.dot(X_, self.weights))
        total_gradient = np.zeros(len(self.weights))

        if max(0, hinge_distance[0]) == 0:
            total_gradient += self.weights
        else:
            total_gradient += self.weights - (self.C * Y[0] * X_[0])

        return total_gradient

    def compute_loss(self, X, Y):
        # Regularization term
        reg_loss = 0.5 * np.dot(self.weights, self.weights)
        # Hinge loss
        distances = 1 - (Y.flatten() * np.dot(X, self.weights))
        distances[distances < 0] = 0
        hinge_loss = self.C * np.mean(distances)
        # Total loss
        loss = reg_loss + hinge_loss
        return loss

    def stochastic_Gradient_Descent(self, X, Y, X_val, Y_val):
        prev_loss = float('inf')
        loss_threshold = 10**(-5)
        train_losses = []
        val_losses = []
        epochs_list = []
        early_stop_epoch = None

        for epoch in range(self.epoch):
            features, output = shuffle(X, Y)

            for i, feature in enumerate(features):
                gradient = self.compute_gradient(feature, output[i])
                self.weights = self.weights - (self.learning_rate * gradient)

            # Compute training and validation loss
            loss = self.compute_loss(X, Y)
            val_loss = self.compute_loss(X_val, Y_val)
            train_losses.append(loss)
            val_losses.append(val_loss)
            epochs_list.append(epoch)

            # Early stopping condition
            if abs(prev_loss - loss) < loss_threshold and early_stop_epoch is None:
                early_stop_epoch = epoch
                print("Early Stopping occurs at 'epoch': {}".format(epoch))
                print("The minimum # of iterations taken:", epoch)
                # Terminate training
                break

            prev_loss = loss

            # Print every 1/10th of the epoch
            if epoch % (self.epoch // 10) == 0:
                print(f"Epoch: {epoch}")
                print("Training Loss: {:.4f},  Validation Loss: {:.4f}".format(loss, val_loss))
                print("------------------------------------------------")

        print("Training ended...")
        print("Weights are: {}".format(self.weights))

        # Plot training and validation loss
        plt.figure()
        plt.plot(epochs_list, train_losses, color='m', label='Training Loss')
        plt.plot(epochs_list, val_losses, color='c', label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()
        
        return epochs_list, train_losses, val_losses


    def mini_Batch_Gradient_Descent(self, X, Y, X_val, Y_val, batch_size, plot=True):
        prev_loss = float('inf')
        loss_threshold = 10**(-5)
        train_losses = []
        val_losses = []
        epochs_list = []
        early_stop_epoch = None
        num_batches = int(np.ceil(len(Y) / batch_size))

        for epoch in range(self.epoch):
            features, output = shuffle(X, Y)
            batches_X = np.array_split(features, num_batches)
            batches_Y = np.array_split(output, num_batches)

            for batch_X, batch_Y in zip(batches_X, batches_Y):
                gradient = np.zeros(len(self.weights))
                for i in range(len(batch_Y)):
                    grad = self.compute_gradient(batch_X[i], batch_Y[i])
                    gradient += grad
                gradient /= len(batch_Y)
                self.weights = self.weights - (self.learning_rate * gradient)

            # Compute training and validation loss
            loss = self.compute_loss(X, Y)
            val_loss = self.compute_loss(X_val, Y_val)
            train_losses.append(loss)
            val_losses.append(val_loss)
            epochs_list.append(epoch)

            # Early stopping condition
            if abs(prev_loss - loss) < loss_threshold and early_stop_epoch is None:
                early_stop_epoch = epoch
                print("Early Stopping occurs at 'epoch': {}".format(epoch))
                print("The minimum # of iterations taken:", epoch)
                break

            prev_loss = loss

            # Print every 1/10th of the epoch
            if epoch % max(1, (self.epoch // 10)) == 0:
                print(f"Epoch: {epoch}")
                print("Training Loss: {:.4f},  Validation Loss: {:.4f}".format(loss, val_loss))
                print("------------------------------------------------")

        print("Training ended...")
        print("Weights are: {}".format(self.weights))

        if (plot):
            # Plot training and validation loss
            plt.figure()
            plt.plot(epochs_list, train_losses, color='m', label='Training Loss (Mini-batch)')
            plt.plot(epochs_list, val_losses, color='c', label='Validation Loss (Mini-batch)')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.legend()
            plt.show()

        return epochs_list, train_losses, val_losses

    def predict(self, X_test, Y_test):
        # Compute predictions on test set
        predicted_values = [np.sign(np.dot(X_test[i], self.weights)) for i in range(X_test.shape[0])]

        # Convert predictions to -1 and 1
        predicted_values = np.array(predicted_values)
        predicted_values[predicted_values == 0] = 1

        # Compute accuracy, precision, and recall
        accuracy = accuracy_score(Y_test, predicted_values)
        precision = precision_score(Y_test, predicted_values, pos_label=1)
        recall = recall_score(Y_test, predicted_values, pos_label=1)
        print("Accuracy: {:.4f} | Precision: {:.4f} | Recall: {:.4f}".format(accuracy, precision, recall))


def part_1(X_train, y_train, X_val, y_val):
    # Model parameters
    C = 0.1
    learning_rate = 0.00001
    epoch = 1000
  
    # Instantiate the SVM class
    my_svm = svm_(learning_rate=learning_rate, epoch=epoch, C_value=C, X=X_train, Y=y_train)

    # Train the model with early stopping
    epochs_sgd, train_losses_sgd, val_losses_sgd = my_svm.stochastic_Gradient_Descent(X_train, y_train, X_val, y_val)

    # Evaluate on validation set
    print("Model performance on validation set (Stochastic Gradient Descent):")
    my_svm.predict(X_val, y_val)

    return my_svm, epochs_sgd, train_losses_sgd, val_losses_sgd


def part_2(X_train, y_train, X_val, y_val):
    # Model parameters
    C = 0.1
    learning_rate = 0.0001  # Increased learning rate
    epoch = 1000
    batch_size = 32

    # Instantiate the SVM class
    my_svm_mini_batch = svm_(learning_rate=learning_rate, epoch=epoch, C_value=C, X=X_train, Y=y_train)

    # Train the model with mini-batch gradient descent
    epochs_mb, train_losses_mb, val_losses_mb = my_svm_mini_batch.mini_Batch_Gradient_Descent(X_train, y_train, X_val, y_val, batch_size)

    # Evaluate on validation set
    print("Model performance on validation set (Stochastic Gradient Descent):")
    my_svm_mini_batch.predict(X_val, y_val)

    return my_svm_mini_batch, epochs_mb, train_losses_mb, val_losses_mb


def part_3(X_train, y_train, X_val, y_val, X_test, y_test, initial_samples=10, max_iterations=100):
    # Parameters for the SVM model
    C = 0.1
    learning_rate = 0.00001
    epoch = 1000
    batch_size = 2
    
    # Step 1: Initialize classifier with a few labeled samples
    X_initial, y_initial = X_train[:initial_samples], y_train[:initial_samples]
    X_pool, y_pool = X_train[initial_samples:], y_train[initial_samples:]

    # Instantiate the SVM class
    active_svm = svm_(learning_rate=learning_rate, epoch=epoch, C_value=C, X=X_initial, Y=y_initial)
    train_losses, val_losses = [], []
    sample_counts = []

    for iteration in range(max_iterations):
        # Train classifier on the current subset
        active_svm.mini_Batch_Gradient_Descent(X_initial, y_initial, X_val, y_val, batch_size, plot = False)

        # Calculate performance metrics on validation set
        val_loss = active_svm.compute_loss(X_val, y_val)
        train_loss = active_svm.compute_loss(X_initial, y_initial)
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        sample_counts.append(len(X_initial))

        # Predict on the remaining pool samples to find the next best sample
        # Stop if there are no more samples in the pool
        if len(X_pool) == 0:
            break  

        pool_losses = np.array([active_svm.compute_loss(X_pool[i:i+1], y_pool[i:i+1]) for i in range(len(X_pool))])
        next_sample_index = np.argmin(pool_losses)  
        # Select the sample with the least loss

        # Add the selected sample to the training set
        X_next_sample = X_pool[next_sample_index:next_sample_index+1]
        y_next_sample = y_pool[next_sample_index:next_sample_index+1]
        X_initial = np.vstack([X_initial, X_next_sample])
        y_initial = np.vstack([y_initial, y_next_sample])

        # Remove the selected sample from the pool
        X_pool = np.delete(X_pool, next_sample_index, axis=0)
        y_pool = np.delete(y_pool, next_sample_index, axis=0)

        # Stop condition: Check if performance on validation set has plateaued
        if iteration > 10 and abs(val_losses[-1] - val_losses[-10]) < 0.001:
            print(f"Early stopping at iteration {iteration}")
            break

        print(f"Iteration {iteration}: Train Loss = {train_loss}, Validation Loss = {val_loss}")

    # Evaluate on test set
    print("Model performance on test set (Active Learning):")
    active_svm.predict(X_test, y_test)
    
    return active_svm, train_losses, val_losses, sample_counts



# Load dataset
print("Loading dataset...")
data = pd.read_csv('data1.csv')

# Drop first and last columns (ID and unnamed columns)
data.drop(data.columns[[-1, 0]], axis=1, inplace=True)

# Segregate inputs and targets
X = data.iloc[:, 1:]
# Add column for bias
X.insert(loc=len(X.columns), column="bias", value=1)
X_features = X.to_numpy()

# Convert categorical variables to integers
category_dict = {'B': -1.0, 'M': 1.0}
Y = np.array([(data.loc[:, 'diagnosis']).to_numpy()]).T
Y_target = np.vectorize(category_dict.get)(Y)
X_train, X_test, y_train, y_test = train_test_split(X_features, Y_target, test_size=0.2, random_state=42)

# Split data into train and test sets
print("Splitting dataset into train and test sets...")
X_train_new, X_val, y_train_new, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# Normalize the data
scaler = StandardScaler().fit(X_train_new)
X_train_scaled = scaler.transform(X_train_new)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)

# Part I Execution
my_svm, epochs_sgd, train_losses_sgd, val_losses_sgd = part_1(X_train_scaled, y_train_new, X_val_scaled, y_val)

# Part II Execution
my_svm_mini_batch, epochs_mb, train_losses_mb, val_losses_mb = part_2(X_train_scaled, y_train_new, X_val_scaled, y_val)

# Plotting the training and validation losses for both methods
plt.figure(figsize=(12, 6))

# Training Losses
plt.subplot(1, 2, 1)
plt.plot(epochs_sgd, train_losses_sgd, color='r', label='Training Loss (Stochastic GD)')
plt.plot(epochs_mb, train_losses_mb, color='b', label='Training Loss (Mini-batch GD)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Comparison')
plt.legend()

# Validation Losses
plt.subplot(1, 2, 2)
plt.plot(epochs_sgd, val_losses_sgd, color='r', label='Validation Loss (Stochastic GD)')
plt.plot(epochs_mb, val_losses_mb, color='b', label='Validation Loss (Mini-batch GD)')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss Comparison')
plt.legend()
plt.tight_layout()
plt.show() 

# Part III Execution
print("Executing Part III: Active Learning")
active_svm, train_losses_al, val_losses_al, sample_counts_al = part_3(X_train_scaled, y_train_new, X_val_scaled, y_val, X_test_scaled, y_test)

# Plotting loss progression for Part III
plt.figure(figsize=(10, 5))
plt.plot(sample_counts_al, train_losses_al, color='m', label='Training Loss (Active Learning)')
plt.plot(sample_counts_al, val_losses_al, color='c', label='Validation Loss (Active Learning)')
plt.xlabel('Number of Training Samples')
plt.ylabel('Loss')
plt.title('Loss Progression in Active Learning (Part III)')
plt.legend()
plt.show()