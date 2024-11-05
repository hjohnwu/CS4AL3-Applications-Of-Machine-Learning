import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn
import sklearn

class my_svm():
    # __init__() function should initialize all your variables
    def __init__(self,):
        pass

    # preprocess() function:
    #  1) normalizes the data, 
    #  2) removes missing values
    #  3) assign labels to target 
    def preprocess(self, data_set):
        return 0
    
    # feature_creation() function takes as input the features set label (e.g. FS-I, FS-II, FS-III, FS-IV)
    # and creates 2 D array of corresponding features 
    # for both positive and negative observations.
    # this array will be input to the svm model
    # For instance, if the input is FS-I, the output is a 2-d array with features corresponding to 
    # FS-I for both negative and positive class observations
    def feature_creation(self, fs_value, data_set):
        match data_set:
            case 'data-2010-15':        
                pos_feaures_main_timechange = np.load("data-2010-15\data-2010-15\\pos_features_main_timechange.npy")
                neg_feaures_main_timechange = np.load("data-2010-15\data-2010-15\\neg_features_main_timechange.npy")
                pos_feaures_historical = np.load("data-2010-15\data-2010-15\\pos_features_historical.npy")
                neg_feaures_historical = np.load("data-2010-15\data-2010-15\\neg_features_historical.npy")
                pos_features_maxmin = np.load("data-2010-15\data-2010-15\\pos_features_maxmin.npy")
                neg_features_maxmin = np.load("data-2010-15\data-2010-15\\neg_features_maxmin.npy")
            case 'data-2020-24':
                pos_feaures_main_timechange = np.load("data-2020-24\data-2020-24\\pos_features_main_timechange.npy")
                neg_feaures_main_timechange = np.load("data-2020-24\data-2020-24\\neg_features_main_timechange.npy")
                pos_feaures_historical = np.load("data-2020-24\data-2020-24\\pos_features_historical.npy")
                neg_feaures_historical = np.load("data-2020-24\data-2020-24\\neg_features_historical.npy")
                pos_features_maxmin = np.load("data-2020-24\data-2020-24\\pos_features_maxmin.npy")
                neg_features_maxmin = np.load("data-2020-24\data-2020-24\\neg_features_maxmin.npy")
            case _:
                return "Unknown data set specified"
        
        #FS-I features:
        FS_1_features_pos = pos_feaures_main_timechange[:, 0:18]
        FS_1_features_neg = neg_feaures_main_timechange[:, 0:18]
        FS_1_features = np.vstack((FS_1_features_pos, FS_1_features_neg))
        FS_1_labels = np.hstack((np.ones(len(FS_1_features_pos)), 0 * np.ones(len(FS_1_features_neg))))
        
        #FS-II features:
        FS_2_features_pos = pos_feaures_main_timechange[:, 18:90]
        FS_2_features_neg = neg_feaures_main_timechange[:, 18:90]
        FS_2_features = np.vstack((FS_2_features_pos, FS_2_features_neg))
        FS_2_labels = np.hstack((np.ones(len(FS_2_features_pos)), 0 * np.ones(len(FS_2_features_neg))))
        
        #FS-III features:
        FS_3_features_pos = pos_feaures_historical[0]
        FS_3_features_neg = neg_feaures_historical[0]
        FS_3_features = np.vstack((FS_3_features_pos, FS_3_features_neg))
        FS_3_labels = np.hstack((np.ones(len(FS_3_features_pos)), 0 * np.ones(len(FS_3_features_neg))))
        
        #FS-IV features:
        FS_4_features_pos = pos_features_maxmin[:, 0:18]
        FS_4_features_neg = neg_features_maxmin[:, 0:18]
        FS_4_features = np.vstack((FS_4_features_pos, FS_4_features_neg))
        FS_4_labels = np.hstack((np.ones(len(FS_4_features_pos)), 0 * np.ones(len(FS_4_features_neg))))
        
        
        FS_features = []
        FS_labels = []
        for fs in fs_value:
            match fs:
                case 'FS-I':
                    return 0
                case 'FS-II':
                    return 0
                case 'FS-II':
                    return 0
                case 'FS-III':
                    return 0
                case 'FS-IV':
                    return 0
                case _:
                    return (f"specified input FS = {fs} is not recognized")
        
        print(f"FS-1 Features = \n {FS_1_features} \n")
        print(f"FS-1 Labels = \n {FS_1_labels} \n")
        print(f"FS-2 Features = \n {FS_2_features} \n")
        print(f"FS-2 Labels = \n {FS_2_labels} \n")
        print(f"FS-3 Features = \n {FS_3_features} \n")
        print(f"FS-3 Labels = \n {FS_3_labels} \n")
        print(f"FS-4 Features = \n {FS_4_features} \n")
        print(f"FS-4 Labels = \n {FS_4_labels} \n")
        
    
    # cross_validation() function splits the data into train and test splits,
    # Use k-fold with k=10
    # the svm is trained on training set and tested on test set
    # the output is the average accuracy across all train test splits.
    def cross_validation(self,):
        # call training function
        # call tss function
        return 0
    
    #training() function trains a SVM classification model on input features and corresponding target
    def training(self, ):
        return 0
    
    # tss() function computes the accuracy of predicted outputs (i.e target prediction on test set)
    # using the TSS measure given in the document
    def tss(self,):
        return 0
    

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
    return 0

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
    return 0

# below should be your code to call the above classes and functions
# with various combinations of feature sets
# and both datasets

feature_experiment()
data_experiment()

svm = my_svm()
features, target = svm.feature_creation(['FS-I'], 'data-2010-15')









        