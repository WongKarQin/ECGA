import numpy as np
import pandas as pd
import time
import sys
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
str_list = ['KNN']
# load data set
raw_data = pd.read_csv('../data_new/duplicate_Mendeley_copy.txt', header=None)
data = raw_data.values
features = data[::, :-1]
labels = data[::, -1]
opts = {}
# Number of instances
num_train = np.size(features, 0)
X = features
y = labels.reshape(num_train)
for classifer_name in str_list:
	opts['classifer'] = classifer_name
	with open('../result/calculation_statistic.txt', 'a') as f:  # 设置文件对象
		f.write(
			'\n' + 'WFS_Mendeley_duplicate_K_fold_' + classifer_name + ' iter_100_size_100' + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_F1.txt', 'a') as f:  # 设置文件对象
		f.write(
			'\n' + 'WFS_Mendeley_duplicate_K_fold_' + classifer_name + ' iter_100_size_100' + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_num_Feature.txt', 'a') as f:  # 设置文件对象
		f.write(
			'\n' + 'WFS_Mendeley_duplicate_K_fold_' + classifer_name + ' iter_100_size_100' + ',')  # 将字符串写入文件中
		f.close()
	for i in range(0, 10):
		time_1 = time.time()
		#random split data into train & validation (3/2 -- 3/1)
		#xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.33, stratify=labels)
		## Number of instances
		#num_train = np.size(xtrain, 0)
		#num_valid = np.size(xtest, 0)
		#ytrain = ytrain.reshape(num_train)  # Solve bug
		#yvalid = ytest.reshape(num_valid)  # Solve bug
		# Training
		if classifer_name == 'CART':
			mdl = DecisionTreeClassifier(criterion='gini')
			#mdl.fit(xtrain, ytrain)
		elif classifer_name == 'KNN':
			mdl = KNeighborsClassifier()
			#mdl.fit(xtrain, ytrain)
		else:
			sys.exit('miss classifer')
		scores = cross_val_score(mdl, X, y, cv=30, scoring='f1')
		F1_score = np.mean(scores)
		#ypred = mdl.predict(xtest)
		#F1_score = f1_score(yvalid, ypred)
		num_feat = len(features[1])
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