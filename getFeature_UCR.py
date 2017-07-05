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
import sys

dataset = 'ECG5000'
# dataType : TEST | TRAIN
# datatype = sys.argv[1]
data = dataloader(True)
df, y, cut_point = data.loadDataForUCR(dataset)
extraction_settings = ComprehensiveFCParameters()
# X = extract_features(df, 
#                      column_id='id', column_sort='time',
#                      default_fc_parameters=extraction_settings,
#                      impute_function= impute)
X_filtered = extract_relevant_features(df, y,
					column_id='id', column_sort='time',
					default_fc_parameters=extraction_settings)
# change column items ',' -> '_'
X_filtered.rename(columns = lambda x:x.replace(',', '_'), inplace=True)
X_filtered.rename(columns = lambda x:x.replace('\"', ''), inplace=True)

X_train, X_test = X_filtered[:cut_point], X_filtered[cut_point:] 
X_train.to_csv('result/' + dataset + '_features_TRAIN.csv')
X_test.to_csv('result/' + dataset + '_features_TEST.csv')
# print(X_filtered.info)