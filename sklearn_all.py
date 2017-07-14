import os
import pandas as pd
from tsfresh import extract_features, extract_relevant_features, select_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import ComprehensiveFCParameters
from sklearn.decomposition import PCA 
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from result_figure import *
# Classifiers
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


# dataset_path = "/home/allen/Code/UCR_TS_Archive_2015"
dataset_path = os.path.expanduser("~/Code/UCR_TS_Archive_2015")

def get_train_test_data(train_origin_file, test_origin_file):
	id_to_target = {}
	df_rows = []
	cur_id = 0
	with open(train_origin_file) as f:	
		for line in f.readlines():
			time = 0
			cur_id += 1
			values = line.split(',')
			id_to_target[cur_id] = int(values[0])
			for value in values[1:]:
				if value not in ['\t', '\n']:
					df_rows.append([cur_id, time, float(value)])
					time += 1
	cut_point = cur_id
	with open(test_origin_file) as t:
		for line in t.readlines():
			time = 0
			cur_id += 1
			values = line.split(',')
			id_to_target[cur_id] = int(values[0])
			for value in values[1:]:
				if value not in ['\t', '\n']:
					df_rows.append([cur_id, time,float(value)])
					time += 1

	df_all = pd.DataFrame(df_rows, columns = (['id', 'time', 'act']))
	y_all = pd.Series(id_to_target)

	return df_all, y_all, cut_point

def get_train_test_feature(data_name, data_all, y_all):
	train_feature_file = 'result/features/' + data_name + '_feature_train.csv'
	test_feature_file = 'result/features/' + data_name + '_feature_test.csv'
	all_feature_file = 'result/features/' + data_name + '_feature_all.csv'
	# get feature from train data
	if os.path.exists(all_feature_file):
		# load feature from local
		with open(all_feature_file) as f:
			all_column = f.readline().split(',')
			all_column[-1] = all_column[-1].strip()
			all_feature = pd.read_csv(all_feature_file, names = all_column, skipinitialspace = True, skiprows = 1)
	else:
		all_train = data_all
		all_y = y_all
		extraction_settings = ComprehensiveFCParameters()
		# all_feature = extract_relevant_features(all_train, all_y,
		# 				column_id='id', column_sort='time',
		# 				default_fc_parameters=extraction_settings)
		all_feature = extract_features(all_train,
						column_id='id', column_sort='time')
		impute(all_feature)
		# change column items ',' -> '_'
		all_feature.rename(columns = lambda x:x.replace(',', '_'), inplace=True)
		all_feature.rename(columns = lambda x:x.replace('\"', ''), inplace=True)
		all_feature.to_csv(all_feature_file)
		all_column = all_feature.columns.tolist()
		all_column[-1] = all_column[-1].strip()



	return all_feature, all_column

def feature_PCA(features, num, data_name):
	_,N = features.shape
	pca_file = 'result/pca/' + data_name + '_pca.csv'
	print('N = ' + str(N))
	if N <= num:
		features.to_csv(pca_file)		
		return features;
	if os.path.exists(pca_file):
		df = pd.read_csv(pca_file)
	else:
		data = features.as_matrix()
		pca = PCA(n_components=num)
		newData = pca.fit_transform(data)
		feature_column = [str(i) for i in range(num)]
		df = pd.DataFrame(data = newData, columns = feature_column)
		df.to_csv(pca_file)

	return df;


def train_and_test(data_name, X_train, Y_train, X_test, Y_test):
	sc = StandardScaler()
	sc.fit(X_train)
	X_train_std = sc.transform(X_train)
	X_test_std = sc.transform(X_test)

	classifiers = []
	accuracy = []
	# ada boost
	clf = AdaBoostClassifier()
	classifiers.append(clf)
	# ann
	clf = MLPClassifier()
	classifiers.append(clf)
	# decision tree
	clf = DecisionTreeClassifier()
	classifiers.append(clf)
	# knn
	clf = KNeighborsClassifier()
	classifiers.append(clf)
	# linear SVM
	clf = SVC()
	classifiers.append(clf)
	# logistic regression
	clf = LogisticRegression()
	classifiers.append(clf)
	# random forrest
	clf = RandomForestClassifier()
	classifiers.append(clf)

	# train and get accuracy
	for clf in classifiers:
		clf.fit(X_train_std, Y_train)
		Z = clf.predict(X_test_std)
		accuracy.append(accuracy_score(Y_test, Z))

	return accuracy

def cross_validation(data_pca, target, fold_index, fold_num, data_name):
	print('data: ' + data_name + ', fold: ' + str(fold_index) + '/' + str(fold_num))
	print(data_pca.shape[0])
	fold_size = int(data_pca.shape[0]/fold_num)
	index_start = fold_index * fold_size
	index_end = index_start + fold_size
	# pd.concat([data_pca[:index_start], data_pca[index_end:]]).to_csv('train.csv')
	# (data_pca[index_start:index_end]).to_csv('test.csv')
	# split train and test
	train_data = pd.concat([data_pca[:index_start], data_pca[index_end:]]).values
	test_data = (data_pca[index_start:index_end]).values
	train_y = pd.concat([target[:index_start], target[index_end:]]).values
	test_y = (target[index_start:index_end]).values
	accuracy = train_and_test(data_name, train_data, train_y, test_data, test_y)

	return accuracy

def divide_and_train(data_name, data_pca, target, cut_point):
	# divide train and test
	train_data = (data_pca[:cut_point]).values
	test_data = data_pca[cut_point:].values
	train_y = target[:cut_point].values
	test_y = target[cut_point:].values
	accuracy = train_and_test(data_name, train_data, train_y, test_data, test_y)
	return accuracy

def get_classified_result(data):
	# load data from dataset
	train_origin_file = os.path.join(dataset_path, data, data + '_TRAIN')
	test_origin_file = os.path.join(dataset_path, data, data + '_TEST')
	result_file = 'result/accuracy/' + data + '_accuracy.csv'
	data_all, y_all, cut_point = get_train_test_data(train_origin_file, test_origin_file)

	# get train and test features
	feature_all, column_all = get_train_test_feature(data, data_all, y_all)

	# PCA
	data_PCA = feature_PCA(feature_all, 20, data)

	# 10-fold Cross Validation
	fold_num = 10
	result = []
	# for fold_index in range(fold_num):
	# 	acc_result = cross_validation(data_PCA, y_all, fold_index, fold_num, data)
	# 	result.append(acc_result)

	acc_result = divide_and_train(data, data_PCA, y_all, cut_point)
	result.append(acc_result)
	df = pd.DataFrame(result, columns = ['adaboost', 'ann', 'decision tree', 'knn', 'linearSVM', 'logistic regression', 'random forrest'])
	df.to_csv(result_file)
	











for data in os.listdir(dataset_path):
# for data in ['ElectricDevices']:
	if data not in ['.DS_Store']:
		print(data)
		get_classified_result(data)