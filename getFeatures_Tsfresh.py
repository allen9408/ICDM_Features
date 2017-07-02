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


def load_data():
	module_path = os.path.dirname(__file__)
	data_file_name = os.path.join(module_path, 'data/real_sequence.txt')

	if not os.path.exists(data_file_name):
		raise RuntimeError('Data file not exist!')

	id_to_target = {}
	df_rows = []

	with open(data_file_name) as f:
		cur_id = 0
		time = 0

		for line in f.readlines():
			cur_id += 1
			time = 0
			id_to_target[cur_id] = 0

			values = line.split(' ')
			for value in values:
				if value not in ['\t', '\n', '101']:
					print([cur_id, time, int(value)])
					df_rows.append([cur_id, time, int(value)])
					time += 1

	df = pd.DataFrame(df_rows, columns=['id', 'time', 'act'])
	y = pd.Series(id_to_target)

	return df, y




df, y = load_data()
extraction_settings = ComprehensiveFCParameters()
X = extract_features(df, 
                     column_id='id', column_sort='time',
                     default_fc_parameters=extraction_settings,
                     impute_function= impute)
X.to_csv('result.csv')