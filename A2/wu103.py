# Credit for some sample/template/pilot code taken from course + lecture content:
# Source: python Assignment_2-starter.py, python support_vector_machine.py ('Swati Mishra', Sep 23, 2024)
# Purpose: This python includes Support Vector Machine Implementation
# License: MIT License
#--------------------------------------------------
# Purpose: Assignment II - Classification
# Author: John Wu
# Student Number: WU103
# Created: Oct 10, 2024
#--------------------------------------------------

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import sklearn.preprocessing
import sklearn.svm as SVM
import sklearn.metrics as Metrics
import sklearn.model_selection as Selection


class my_svm():
    # __init__() function should initialize all your variables
    def __init__(self, data_set):
        
        # Load data according to specified data set and its path
        # Load the data as numpy arrays and convert to DataFrames
        self.pos_feaures_main_timechange = pd.DataFrame(np.load(f"{data_set}\\{data_set}\\pos_features_main_timechange.npy"))
        self.neg_feaures_main_timechange = pd.DataFrame(np.load(f"{data_set}\\{data_set}\\neg_features_main_timechange.npy"))
        self.pos_feaures_historical = pd.DataFrame(np.load(f"{data_set}\\{data_set}\\pos_features_historical.npy"))
        self.neg_feaures_historical = pd.DataFrame(np.load(f"{data_set}\\{data_set}\\neg_features_historical.npy"))
        self.pos_features_maxmin = pd.DataFrame(np.load(f"{data_set}\\{data_set}\\pos_features_maxmin.npy"))
        self.neg_features_maxmin = pd.DataFrame(np.load(f"{data_set}\\{data_set}\\neg_features_maxmin.npy"))
        # Load the shuffled index order (data_order.npy)
        self.data_order = np.load(f"{data_set}\\{data_set}\\data_order.npy")
        
        self.FS_1_features = None
        self.FS_2_features = None
        self.FS_3_features = None
        self.FS_4_features = None
        self.labels = None
        self.X_features = None
        self.Y_labels = None
        self.model = None
        self.scaler = sklearn.preprocessing.StandardScaler()


    # preprocess() function:
    # 1) Normalizes the data
    # 2) Removes missing values
    # 3) Applies shuffled indices from `data_order.npy` to shuffle both features and labels
    def preprocess(self):
        
        # Concatenate all positive and negative features as pandas DataFrames
        all_pos_features = pd.concat([self.pos_feaures_main_timechange, self.pos_feaures_historical, self.pos_features_maxmin], axis=1)
        all_neg_features = pd.concat([self.neg_feaures_main_timechange, self.neg_feaures_historical, self.neg_features_maxmin], axis=1)
        
        # Combine all features into one DataFrame
        all_features = pd.concat([all_pos_features, all_neg_features], axis=0).reset_index(drop=True)
        
        # Generate labels: 1 for positives, 0 for negatives (same for all feature sets)
        self.labels = pd.Series([1] * len(all_pos_features) + [0] * len(all_neg_features))

        # Apply the shuffling order from `data_order.npy`
        # Reduces any potential bias that having 2 splits of stacked positive (1) or negative (0) rows may have on training
        all_features = all_features.iloc[self.data_order].reset_index(drop=True)
        self.labels = self.labels.iloc[self.data_order].reset_index(drop=True)

        # Apply scaling and normalization to the entire DataFrame
        all_features_scaled = pd.DataFrame(self.scaler.fit_transform(all_features), columns=all_features.columns)
        
        # Split scaled data back into individual feature sets (FS-1, FS-2, FS-3, FS-4)
        self.FS_1_features = all_features_scaled.iloc[:, 0:18]
        self.FS_2_features = all_features_scaled.iloc[:, 18:90]
        self.FS_3_features = all_features_scaled.iloc[:, 90:91]
        self.FS_4_features = all_features_scaled.iloc[:, 91:109]


    # feature_creation() function takes as input the features set label (e.g. FS-I, FS-II, FS-III, FS-IV)
    # and creates 2D DataFrame of corresponding features for both positive and negative observations.
    def feature_creation(self, fs_value):
        features = []
        for fs in fs_value:
            match fs:
                case 'FS-I':
                    if self.FS_1_features is not None:
                        features.append(self.FS_1_features)
                case 'FS-II':
                    if self.FS_2_features is not None:
                        features.append(self.FS_2_features)
                case 'FS-III':
                    if self.FS_3_features is not None:
                        features.append(self.FS_3_features)
                case 'FS-IV':
                    if self.FS_4_features is not None:
                        features.append(self.FS_4_features)
                case _:
                    return f"Specified input FS = {fs} is not recognized"
        
        # Concatenate all selected features into a single DataFrame
        if len(features) != 0:
            final_features = pd.concat(features, axis=1)
            return final_features, self.labels
        else:
            return None, None


    # cross_validation() function: K-fold cross-validation process for calculating TSS.
    def cross_validation(self, X, Y, k=10, combination_name="Feature Set"):
        
        # Initialize KFold with k splits
        kf = Selection.KFold(n_splits=k, shuffle=True, random_state=42)

        # List to store TSS scores for each fold
        tss_scores = []

        # Create a GridSpec layout for confusion matrices and TSS plot side by side
        fig = plt.figure(figsize=(15, 5))
        fig.suptitle(f"Confusion Matrices and TSS Scores for {combination_name} (10 Folds)", fontsize=16)

        # Use GridSpec to allocate more space for confusion matrices and the TSS plot to the right
        # 5 columns for confusion matrices, 1 wider column for TSS
        gs = gridspec.GridSpec(2, 6, width_ratios=[1, 1, 1, 1, 1, 2])

        # K-fold cross-validation process
        for i, (train_index, test_index) in enumerate(kf.split(X)):
            
            # Split the data into training and test sets
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            Y_train, Y_test = Y.iloc[train_index], Y.iloc[test_index]

            # Normalize the training and test sets separately to avoid data leakage
            X_train = pd.DataFrame(self.scaler.fit_transform(X_train), columns=X_train.columns)
            X_test = pd.DataFrame(self.scaler.transform(X_test), columns=X_test.columns)

            # Train the model on the training set
            self.training(X_train, Y_train)

            # Predict on the test set
            Y_pred = self.model.predict(X_test)

            # Calculate TSS for the current fold
            tss_score = self.tss(Y_test, Y_pred)
            tss_scores.append(tss_score)

            # Calculate the confusion matrix
            confusion_matrix = Metrics.confusion_matrix(Y_test, Y_pred)

            # Plot the confusion matrix on the corresponding subplot
            # Use GridSpec for layout (2 rows, 5 columns)
            ax = fig.add_subplot(gs[i // 5, i % 5])
            sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="viridis", cbar=False, ax=ax)
            ax.set_title(f"Fold {i + 1}")

            # Set axis titles
            ax.set_xlabel('Predicted label')
            ax.set_ylabel('True label')

        # Plot the TSS scores across the folds in the last column
        # The entire right column for TSS line plot
        tss_ax = fig.add_subplot(gs[:, 5])
        sns.lineplot(x=range(1, k + 1), y=tss_scores, marker='o', ax=tss_ax)
        tss_ax.set_xlabel('Fold Number')
        tss_ax.set_ylabel('TSS Score')
        tss_ax.grid(True)

        # Adjust layout and show the combined plot
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show()

        # Calculate and print the average TSS across all folds
        average_tss = np.mean(tss_scores)
        print(f"Average TSS for {combination_name} across {k} folds: {average_tss}")

        return average_tss


    # Training function for SVM
    def training(self, X, Y):
        
        # Some experimenting with different SVM designs and parameters:
        # svm_model = SVM.SVC(kernel='linear', random_state=42)
        # Yeilded best TSS = 0.75     with [FS-IV]
        # svm_model = SVM.SVC(kernel='poly', degree=3, random_state=42)
        # Yeilded best TSS = 0.68     with [FS-IV]
        # svm_model = SVM.SVC(kernel='rbf', random_state=42)
        # Yeilded best TSS = 0.77     with [FS-I, FS-IV]
        # svm_model = SVM.SVC(kernel='rbf', C=10, random_state=42)
        # Yeilded best TSS = 0.78     with [FS-I, FS-IV]     

        # Initialize an SVM model using the SVC methods provided by sklearn
        svm_model = SVM.SVC(kernel='rbf', C=10, gamma='scale', random_state=42)
        
        # Training and fitting our data using the SVM with our Features and Labels
        svm_model.fit(X, Y)
        self.model = svm_model


    # TSS function for performance calculation
    def tss(self, Y_true, Y_pred):
        confusion_matrix = Metrics.confusion_matrix(Y_true, Y_pred)
        TN, FP, FN, TP = confusion_matrix.ravel()
        TPR = TP / (TP + FN) if (TP + FN) > 0 else 0
        FPR = FP / (FP + TN) if (FP + TN) > 0 else 0
        tss_score = TPR - FPR
        return tss_score


# Combination of all the possible feature sets
all_combinations = [
    ['FS-I'],
    ['FS-II'], 
    ['FS-III'],
    ['FS-IV'],
    ['FS-I', 'FS-II'], 
    ['FS-I', 'FS-III'], 
    ['FS-I', 'FS-IV'],
    ['FS-II', 'FS-III'], 
    ['FS-II', 'FS-IV'], 
    ['FS-III', 'FS-IV'],
    ['FS-I', 'FS-II', 'FS-III'], 
    ['FS-I', 'FS-II', 'FS-IV'],
    ['FS-I', 'FS-III', 'FS-IV'], 
    ['FS-II', 'FS-III', 'FS-IV'],
    ['FS-I', 'FS-II', 'FS-III', 'FS-IV']
]

# feature_experiment() function executes experiments with all 4 feature sets.
# svm is trained (and tested) on 2010 dataset with all 4 feature set combinations
# the output of this function is : 
#  1) TSS average scores (mean std) for k-fold validation printed out on console.
#  2) Confusion matrix for TP, FP, TN, FN. See assignment document 
#  3) A chart showing TSS scores for all folds of CV. 
#     This means that for each fold, compute the TSS score on test set for that fold, and plot it.
#     The number of folds will decide the number of points on this chart (i.e 10)
#
# Above 3 charts are produced for all feature combinations
#  4) The function prints the best performing feature set combination
def feature_experiment():
    svm = my_svm('data-2010-15')
    svm.preprocess()

    best_tss = -np.inf
    best_combination = None
    
    for combination in all_combinations:
        print(f"\nEvaluating combination: {combination}\n")
        
        X_features, Y_labels = svm.feature_creation(combination)
        avg_tss = svm.cross_validation(X_features, Y_labels, k=10, combination_name=str(combination))
                
        if avg_tss > best_tss:
            best_tss = avg_tss
            best_combination = combination

    print(f"\nBest feature set combination: {best_combination} with TSS: {best_tss}")


# data_experiment() function executes 2 experiments with 2 data sets.
# svm is trained (and tested) on both 2010 data and 2020 data
# the output of this function is : 
#  1) TSS average scores for k-fold validation printed out on console.
#  2) Confusion matrix for TP, FP, TN, FN. See assignment document 
#  3) A chart showing TSS scores for all folds of CV. 
#     This means that for each fold, compute the TSS score on test set for that fold, and plot it.
#     The number of folds will decide the number of points on this chart (i.e. 10)
# above 3 charts are produced for both datasets
# feature set for this experiment should be the 
# best performing feature set combination from feature_experiment()
def data_experiment():
    svm = my_svm('data-2020-24')
    svm.preprocess()
    
    best_combination = ['FS-I', 'FS-IV']

    print(f"\nEvaluating combination: {best_combination}\n")
    X_features, Y_labels = svm.feature_creation(best_combination)
    avg_tss = svm.cross_validation(X_features, Y_labels, k=10, combination_name=str(best_combination))
    
    print(f"Average TSS for best feature combination on 2020-24 dataset = {avg_tss}")


# below should be your code to call the above classes and functions
# with various combinations of feature sets
# and both datasets

feature_experiment()
data_experiment()