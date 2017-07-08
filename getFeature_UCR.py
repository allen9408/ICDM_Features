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
from featureloader import featureloader
import sys

dataset = 'ECG5000'
feature_file_name = 'result/' + dataset + '_features_ALL.csv'
# dataType : TEST | TRAIN
# datatype = sys.argv[1]
if os.path.exists(feature_file_name):
	# load feature from local
	with open(feature_file_name) as f:
		feature_columns_line = f.readline().split(',')
		feature_columns_line[-1] = feature_columns_line[-1].strip()
		X_filtered = pd.read_csv(feature_file_name, names = feature_columns_line, skipinitialspace = True, skiprows = 1)
else:
	# extract feature from data
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
	X_filtered.to_csv('result/' + dataset + '_features_ALL.csv')
# PCA
fl = featureloader('_', '_')
X_filtered = fl.feature_PCA(X_filtered, 10)

cut_point = 500
X_train, X_test = X_filtered[:cut_point], X_filtered[cut_point:] 
X_train.to_csv('result/' + dataset + '_features_TRAIN.csv')
X_test.to_csv('result/' + dataset + '_features_TEST.csv')
# print(X_filtered.info)