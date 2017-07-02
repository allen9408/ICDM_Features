import matplotlib.pylab as plt
import seaborn as sns
from tsfresh.examples.robot_execution_failures import download_robot_execution_failures, load_robot_execution_failures
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.tree import DecisionTreeClassifier
from sklearn.cross_validation import train_test_split
from sklearn.metrics import classification_report
import os
import pandas as pd
from dataloader import dataloader

dataset = 'ECG5000'
data = dataloader(True)
df, y = data.loadDataForUCR(dataset)
extraction_settings = ComprehensiveFCParameters()
X = extract_features(df, 
                     column_id='id', column_sort='time',
                     default_fc_parameters=extraction_settings,
                     impute_function= impute)
X.to_csv('result/' + dataset + '_features.csv')
print(X.info)