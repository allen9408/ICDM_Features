import itertools
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import pandas as pd
import os
def plot_confusion_matrix(real_vec,pre_vec,classes,normalize=False,title='Confusion Matrix',cmap=plt.cm.Blues):
	# Get confusion matrix
	cm = confusion_matrix(real_vec, pre_vec)

	plt.figure()
	plt.imshow(cm, interpolation='nearest', cmap=cmap)
	plt.title(title)
	plt.colorbar()
	tick_marks = np.arange(len(classes))
	plt.xticks(tick_marks, classes, rotation=45)
	plt.yticks(tick_marks, classes)
	if normalize:
		cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
		print("Normalized confusion matrix")
	else:
		print('Confusion matrix, without normalization')

	print(cm)

	thresh = cm.max() / 2.
	for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
		plt.text(j, i, cm[i, j],
				 horizontalalignment="center",
				 color="white" if cm[i, j] > thresh else "black")

	plt.tight_layout()
	plt.ylabel('True label')
	plt.xlabel('Predicted label')

	plt.show()

def plot_result(dataset, Y, Z, clf, title):
	# load X from dataset
	datafile = 'result/' + dataset + '_plot_TRAIN.csv'
	if not os.path.exists(datafile):
		raise RuntimeError('Data file:' + datafile + ' not exists!')

	X = pd.read_csv(datafile, skipinitialspace = True).values[:, 1:]
	print(X.shape)
	x_min, x_max = X[:, 0].min(), X[:, 0].max()
	y_min, y_max = X[:, 1].min(), X[:, 1].max()

	# h = 0.02

	# xx, yy =  np.meshgrid(np.arange(x_min, x_max, 10), np.arange(y_min, y_max, 10))

	# Z = Z.reshape(xx.shape)

	plt.figure(1, figsize=(4,3))
	# plt.scatter(X[:,1], X[:,0])

	# plt.show()
	color_map = ['b', 'g', 'r', 'c', 'y']
	# f = plt.figure(2)
	# for i in range(1, 6):
	# 	idx = np.where(Z==i)
	# 	plt.scatter(X[idx, 1], X[idx, 0], color = color_map[i-1])
	# plt.contourf(X[:, 0], X[:, 1], Z, cmap=plt.cm.coolwarm, alpha=0.8)
	plt.scatter(X[:, 0], X[:, 1], c = Z, cmap=plt.cm.coolwarm)
	# idx_1 = np.where(Z==1)
	# plt.scatter(X[idx_1, 1], X[idx_1, 0], color = 'm')
	# plt.show()
	plt.savefig('result/im/' + title + '.png')