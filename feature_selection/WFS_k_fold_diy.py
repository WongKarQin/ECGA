import numpy as np
import pandas as pd
import time
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
str_list = ['CART','KNN']
# load data set
raw_data = pd.read_csv('../data_new/UCI.txt', header=None)
data = raw_data.values
features = data[::, :-1]
labels = data[::, -1]
opts={}
num_train = np.size(features, 0)
for classifer_name in str_list:
	opts['classifer'] = classifer_name
	with open('../result/calculation_statistic.txt', 'a') as f:  # 设置文件对象
		f.write(
			'\n' + 'WFS_UCI_original_10_fold_' + classifer_name + ' iter_100_size_100' + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_F1.txt', 'a') as f:  # 设置文件对象
		f.write(
			'\n' + 'WFS_UCI_original_10_fold_' + classifer_name + ' iter_100_size_100' + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_num_Feature.txt', 'a') as f:  # 设置文件对象
		f.write(
			'\n' + 'WFS_UCI_original_10_fold_' + classifer_name + ' iter_100_size_100' + ',')  # 将字符串写入文件中
		f.close()
	for i in range(0, 5):
		time_1 = time.time()
		data_num = np.arange(0, num_train, 1)
		kf = KFold(n_splits=10, shuffle=True)
		d = kf.split(data_num)
		scores = []
		for train_idx, test_idx in d:
			xtrain = features[train_idx]
			ytrain = labels[train_idx]
			xtest = features[test_idx]
			ytest = labels[test_idx]
			if classifer_name == 'CART':
				mdl = DecisionTreeClassifier(criterion='gini')
				mdl.fit(xtrain, ytrain)
			elif classifer_name == 'KNN':
				mdl = KNeighborsClassifier()
				mdl.fit(xtrain, ytrain)
			else:
				sys.exit('miss classifer')
			ypred = mdl.predict(xtest)
			F1_score = f1_score(ytest, ypred)
			if F1_score == 0:
				continue
			else:
				scores.append(F1_score)
		num_feat=len(features[1])
		F1_score = np.mean(scores)
		time_2 = time.time()
		time_cost = time_2 - time_1
		with open('../result/calculation_statistic.txt', 'a') as f:  # 设置文件对象
			f.write(str(time_cost) + ',')  # 将字符串写入文件中
			f.close()
		with open('../result/calculation_F1.txt', 'a') as f:  # 设置文件对象
			f.write(str(F1_score) + ',')  # 将字符串写入文件中
			f.close()
		with open('../result/calculation_num_Feature.txt', 'a') as f:  # 设置文件对象
			f.write(str(num_feat) + ',')  # 将字符串写入文件中
			f.close()