import os
import pandas as pd
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters

# if os.path.exists('result/sklearn_result.csv'):
# 	os.remove('result/sklearn_result.csv')

# os.popen('python linearSVM_sklearn.py')
# os.popen('python logisticRegression_sklearn.py')
# os.popen('python decisionTree_sklearn.py')
# os.popen('python ann_sklearn.py')
# os.popen('python knn_sklearn.py')
# os.popen('python randomForrest_sklearn.py')
# os.popen('python adaboost_sklearn.py')

dataset_path = "/Users/allen/Code/UCR_TS_Archive_2015"

def get_train_test_data(train_origin_file, test_origin_file):
	id_to_target_train = {}
	df_rows_train = []
	with open(train_origin_file) as f:
		cur_id = 0
		for line in f.readlines():
			time = 0
			cur_id += 1
			values = line.split(',')
			for value in values[1:]:
				if value not in ['\t', '\n']:
					df_rows_train.append([cur_id, time, float(value)])
					time += 1

	id_to_target_test = {}
	df_rows_test = []
	with open(test_origin_file) as t:
		cur_id = 0
		for line in t.readlines():
			time = 0
			cur_id += 1
			values = line.split(',')
			id_to_target_test[cur_id] = int(values[0])
			for value in values[1:]:
				if value not in ['\t', '\n']:
					df_rows_test.append([cur_id, time,float(value)])
					time += 1

	df_train = pd.DataFrame(df_rows_train, columns = (['id', 'time', 'act']))
	df_test = pd.DataFrame(df_rows_test, columns = (['id', 'time', 'act']))
	y_train = pd.Series(id_to_target_train)
	y_test = pd.Series(id_to_target_test)

	return df_train, y_train, df_test, y_test

def get_train_test_feature(train_data, train_y, test_data, test_y, data_name):
	train_feature_file = 'result/' + data_name + '_feature_train.csv'
	test_feature_file = 'result/' + data_name + '_feature_test.csv'
	# get feature from train data
	if os.path.exists(train_feature_file):
		# load feature from local
		with open(train_feature_file) as f:
			train_column = f.readlines().split(',')
			train_column[-1] = train_column[-1].strip()
			train_feature = pd.read_csv(train_feature_file, names = train_column, skipinitialspace = True, skiprows = 1)
	else:
		extraction_settings = ComprehensiveFCParameters()
		feature_train = extract_relevant_features(train_data, train_y,
						column_id='id', column_sort='time',
						default_fc_parameters=extraction_settings)
		# change column items ',' -> '_'
		feature_train.rename(columns = lambda x:x.replace(',', '_'), inplace=True)
		feature_train.rename(columns = lambda x:x.replace('\"', ''), inplace=True)
		feature_train.to_csv(train_feature_file)

	# get feature from test data
	if os.path.exists(test_feature_file):
		# load feature from local
		with open(test_feature_file) as f:
			test_column = f.readlines().split(',')
			test_column[-1] = test_column[-1].strip()
			test_feature = pd.read_csv(test_feature_file, names = test_column, skipinitialspace = True, skiprows = 1)
	else:
		extraction_settings = ComprehensiveFCParameters()
		feature_test = extract_relevant_features(test_data, test_y,
						column_id='id', column_sort='time',
						default_fc_parameters=extraction_settings)
		# change column items ',' -> '_'
		feature_test.rename(columns = lambda x:x.replace(',', '_'), inplace=True)
		feature_test.rename(columns = lambda x:x.replace('\"', ''), inplace=True)
		feature_test.to_csv(test_feature_file)

	return feature_train, feature_test, train_column, test_column


def get_classified_result(data):
	# load data from dataset
	train_origin_file = os.path.join(dataset_path, data, data + '_TRAIN')
	test_origin_file = os.path.join(dataset_path, data, data + '_TEST')

	train_data, train_y, test_data, test_y = get_train_test_data(train_origin_file, test_origin_file)

	# get train and test features
	feature_train, feature_test, names_train, names_test = get_train_test_feature(train_data, train_y, test_data, test_y, data)

	print(names_train.shape)
	print(names_test.shape)

	











for data in os.listdir(dataset_path):
	if data not in ['.DS_Store']:
		print(data)
		get_classified_result(data)