# Credit for some sample/template/pilot code taken from course + lecture content:
# Source: python linear_regression.py ('Swati Mishra', Sep 3, 2024)
# Purpose: This python includes OLS method for Linear Regression 
# License: MIT License
#--------------------------------------------------
# Purpose: Assignment 1 - Regression (Part I)
# Author: John Wu
# Student Number: WU103
# Created: Sep 10, 2024
#--------------------------------------------------

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
np.random.seed(42)

# import data
data = pd.read_csv("gdp-vs-happiness.csv")
# drop columns that will not be used
by_year = (data[data['Year']==2018]).drop(columns=["Continent","Population (historical estimates)","Code"])
# remove missing values from columns 
df = by_year[(by_year['Cantril ladder score'].notna()) & (by_year['GDP per capita, PPP (constant 2017 international $)']).notna()]

# create np.array for gdp and happiness where happiness score is above 4.5
happiness=[]
gdp=[]
for row in df.iterrows():
    if row[1]['Cantril ladder score']>4.5:
        happiness.append(row[1]['Cantril ladder score'])
        gdp.append(row[1]['GDP per capita, PPP (constant 2017 international $)'])

class linear_regression():
 
    def __init__(self, x_:list, y_:list) -> None:

        self.input = np.array(x_)
        self.target = np.array(y_)

    def preprocess(self,):

        # normalize the values
        hmean = np.mean(self.input)
        hstd = np.std(self.input)
        x_train = (self.input - hmean)/hstd

        # arrange in matrix format
        X = np.column_stack((np.ones(len(x_train)),x_train))

        # normalize the values
        gmean = np.mean(self.target)
        gstd = np.std(self.target)
        y_train = (self.target - gmean)/gstd

        # arrange in matrix format
        Y = (np.column_stack(y_train)).T

        return X, Y

    def train_ols(self, X, Y):
        # compute and return beta
        return np.linalg.inv(X.T.dot(X)).dot(X.T).dot(Y)
    
    def train_GD(self, X, Y, alpha, epochs):
        # Step 2a: Choose arbitrary value of  𝛽
        beta = np.random.randn(2,1)
        for i in range (epochs):
            # Step 2b: Find the steepest descent ∇𝛽𝐾
            n = len(Y)
            gradients = (2/n) * (X.T).dot(X.dot(beta) - Y)
            # Step 2c: Compute new value of 𝛽𝑛𝑒𝑤 = 𝛽 - α∇𝛽𝐾
            beta = beta - alpha * gradients
            # repeat
        return beta
    
    def predict(self, X_test,beta):
        # predict using beta
        Y_hat = X_test*beta.T
        return np.sum(Y_hat,axis=1)
    
    def MSE(self, Y, Y_hat):
        mse = np.mean((Y - Y_hat) ** 2)
        return mse

# instantiate the linear_regression class  
lr = linear_regression(gdp,happiness)
# preprocess the inputs
X,Y = lr.preprocess()

# For OLS Calculation:
# compute beta
beta_ols = lr.train_ols(X,Y)
# use the computed beta for prediction
Y_predict_ols = lr.predict(X,beta_ols)

# For GD Calculations
def obtain_data_GD(X, Y, alpha, epochs, color):
    # compute beta
    beta_GD = lr.train_GD(X,Y,alpha,epochs)
    # use the computed beta for prediction
    Y_predict_GD = lr.predict(X, beta_GD)
    # store data to graph this line and print the information
    GD_line_data = {
     "alpha":       alpha,
     "epochs":      epochs,
     "beta":        beta_GD,
     "Y_predict":   Y_predict_GD,
     "color":       color   
    }
    return GD_line_data

# Test GD lines, to experiment with various learning rates and epochs
# I explicitly experimented with 5 different alphas and 5 different epochs
# the 8 lines left are the ones submitted onto the graph

#GD_1 = obtain_data_GD(X,Y,0.01,100,'b')
GD_2 = obtain_data_GD(X,Y,0.01,125,'y')
GD_3 = obtain_data_GD(X,Y,0.01,150,'c')
GD_4 = obtain_data_GD(X,Y,0.01,175,'m')
GD_5 = obtain_data_GD(X,Y,0.01,200,'g')

GD_6 = obtain_data_GD(X,Y,0.1,100,'b')
GD_7 = obtain_data_GD(X,Y,0.05,500,'y')
GD_8 = obtain_data_GD(X,Y,0.01,500,'c')
GD_9 = obtain_data_GD(X,Y,0.005,500,'m')
#GD_10 = obtain_data_GD(X,Y,0.001,500,'g')

best_GD = GD_8
array_GD = [GD_2,GD_3,GD_4,GD_5,GD_6,GD_7,GD_8,GD_9]


# below code displays the predicted values


print("\nFOR ASSIGNMENT PART I: Q1)")
# access the 1st column (the 0th column is all 1's)
X_ = X[...,1].ravel()
# set the plot and plot size
fig, ax = plt.subplots()
plt.figure(1)
fig.set_size_inches((15,8))
# display the X and Y points
ax.scatter(X_,Y)
# display the line predicted by beta and X
# Gradient Descent Lines:
for GD in array_GD:
    ax.plot(X_,GD["Y_predict"],color=GD["color"], label=f'GD prediction with "alpha" = {GD["alpha"]} "Epochs" = {GD["epochs"]}')  
    print(f'\nGradient Descent with:\n "alpha" = {GD["alpha"]} "Epochs" = {GD["epochs"]}\n "beta" = {GD["beta"].flatten()}') 
# set the x-labels
ax.set_xlabel("GDP per capita")
# set the x-labels
ax.set_ylabel("Happiness")
# set the title
ax.set_title("GDP per capita of countries (2018) vs Cantril Ladder Score")
# set the legend
ax.legend()


print("\nFOR ASSIGNMENT PART I: Q2)")
# access the 1st column (the 0th column is all 1's)
X_ = X[...,1].ravel()
# set the plot and plot size
fig, ax = plt.subplots()
plt.figure(2)
fig.set_size_inches((15,8))
# display the X and Y points
ax.scatter(X_,Y)
# display the line predicted by beta and X
# best Gradient Descent Line
ax.plot(X_,best_GD["Y_predict"],color=best_GD["color"], label=f'GD prediction with "alpha" = {best_GD["alpha"]} "Epochs" = {best_GD["epochs"]}')  
print(f'\nBEST Gradient Descent with:\n "alpha" = {best_GD["alpha"]} "Epochs" = {best_GD["epochs"]}\n "beta" = {best_GD["beta"].flatten()}')     
# OLS Line
ax.plot(X_,Y_predict_ols,color='r', label="OLS prediction")
print(f'\nOriginal Least Squares (OLS) had\n "beta" = {beta_ols.flatten()}')
# set the x-labels
ax.set_xlabel("GDP per capita")
# set the x-labels
ax.set_ylabel("Happiness")
# set the title
ax.set_title("GDP per capita of countries (2018) vs Cantril Ladder Score")
# set the legend
ax.legend()


# show the plot
plt.show()