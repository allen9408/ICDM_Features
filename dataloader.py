import os
import pandas as pd

class dataloader(object):
	"""Dataloader for Tsfresh"""
	dataLen = 0
	classNum = 0
	needTrain = False
	def __init__(self, needTrain):
		super(dataloader, self).__init__()
		self.needTrain = needTrain
		
	def loadDataForICDM(self):
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

	def loadDataForUCR(self, filename):
		data_path = ('/Users/allen/Code/UCR_TS_Archive_2015')
		data_file_name = os.path.join(data_path, filename, filename + '_TRAIN')
		if not os.path.exists(data_file_name):
			raise RuntimeError('Data file:' + data_file_name + ' not exist!')

		id_to_target = {}
		df_rows = []

		with open(data_file_name) as f:
			cur_id = 0
			for line in f.readlines():
				time = 0
				cur_id += 1
				values = line.split(',')
				self.dataLen = len(values) - 1
				id_to_target[cur_id] = int(values[0])
				for value in values[1:]:
					if value not in ['\t', '\n']:
						df_rows.append([cur_id, time, float(value)])
						time += 1

		cut_point = cur_id
		data_file_name = os.path.join(data_path, filename, filename + '_TEST')
		if not os.path.exists(data_file_name):
			raise RuntimeError('Data file:' + data_file_name + ' not exist!')

		with open(data_file_name) as f:
			for line in f.readlines():
				time = 0
				cur_id += 1
				values = line.split(',')
				self.dataLen = len(values) - 1
				id_to_target[cur_id] = int(values[0])
				for value in values[1:]:
					if value not in ['\t', '\n']:
						df_rows.append([cur_id, time, float(value)])
						time += 1


		df = pd.DataFrame(df_rows, columns=(['id', 'time', 'act']))
		y = pd.Series(id_to_target)

		return df, y, cut_point
