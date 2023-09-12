from ECGA_k_fold import *
import matplotlib.pyplot as plt
import numpy as np1
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import sys
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB, ComplementNB, CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
import time
from sklearn.model_selection import cross_val_score


# 'BNB',
# 'SVM', 'RF'
# str_list = ['CART','MLP','CANB', 'LR', 'KNN', 'SGD']
str_list = ['CART','KNN']
# load data set
raw_data = pd.read_csv('../data_new/UCI.txt', header=None)
data = raw_data.values
features = data[::, :-1]
labels = data[::, -1]
# # split data into train & validation (3/2 -- 3/1)
# xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.33, stratify=labels)
# fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}

# parameters
dim = len(features[1])
number_particles = 100
iter_num = 100
x_max = 10
max_vel = 0.5
opts = {'classifer': '', 'dim': dim, 'N': number_particles, 'T': iter_num, 'x_max': x_max,
        'max_vel': max_vel}
for classifer_name in str_list:
	opts['classifer'] = classifer_name
	with open('../result/calculation_statistic.txt', 'a') as f:  # 设置文件对象
		f.write(
			'\n' + 'ECGA_UCI_original_iterate100_size100_10_fold_' + classifer_name + ' iter_100_size_100' + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_F1.txt', 'a') as f:  # 设置文件对象
		f.write(
			'\n' + 'ECGA_UCI_original_iterate100_size100_10_fold_' + classifer_name + ' iter_100_size_100' + ',')  # 将字符串写入文件中
		f.close()
	# with open('../result/calculation_acc.txt', 'a') as f:  # 设置文件对象
	#	f.write('\n' + 'GA_Mendeley_iterate100_size100_' + classifer_name + ' iter_100_size_100' + ',')  # 将字符串写入文件中
	#	f.close()
	with open('../result/calculation_num_Feature.txt', 'a') as f:  # 设置文件对象
		f.write(
			'\n' + 'ECGA_UCI_original_iterate100_size100_10_fold_' + classifer_name + ' iter_100_size_100' + ',')  # 将字符串写入文件中
		f.close()
	for i in range(0, 5):
		# # split data into train & validation (3/2 -- 3/1)
		# xtrain, xtest, ytrain, ytest = train_test_split(features, labels, test_size=0.33, stratify=labels)
		# fold = {'xt': xtrain, 'yt': ytrain, 'xv': xtest, 'yv': ytest}
		# opts['fold']=fold
		#opts['x'] = features
		#opts['y'] = labels
		time_1 = time.time()
		# perform feature selection
		fmdl = jfs(features, labels, opts)
		sf = fmdl['sf']  # selected features
		
		# # model with selected features
		# num_train = np.size(xtrain, 0)
		# num_valid = np.size(xtest, 0)
		# x_train = xtrain[:, sf]
		# y_train = ytrain.reshape(num_train)  # Solve bug
		# x_valid = xtest[:, sf]
		# y_valid = ytest.reshape(num_valid)  # Solve bug
		#
		# if classifer_name == 'CART':
		# 	mdl = DecisionTreeClassifier(criterion='gini')
		# 	mdl.fit(x_train, y_train)
		# elif classifer_name == 'SVM':
		# 	mdl = svm.SVC()
		# 	mdl.fit(x_train, y_train)
		# elif classifer_name == 'MLP':
		# 	mdl = Perceptron()
		# 	mdl.fit(x_train, y_train)
		# elif classifer_name == 'BNB':
		# 	mdl = BernoulliNB()  # 采用伯努利贝叶斯
		# 	mdl.fit(x_train, y_train)
		# elif classifer_name == 'GNB':
		# 	mdl = GaussianNB()  # 采用高斯贝叶斯
		# 	mdl.fit(x_train, y_train)
		# elif classifer_name == 'MNB':
		# 	mdl = MultinomialNB()  # 采用多项式贝叶斯
		# 	mdl.fit(x_train, y_train)
		# elif classifer_name == 'CONB':
		# 	mdl = ComplementNB()  # 采用互补朴素贝叶斯
		# 	mdl.fit(x_train, y_train)
		# elif classifer_name == 'CANB':
		# 	mdl = CategoricalNB()  # 采用绝对贝叶斯
		# 	mdl.fit(x_train, y_train)
		# elif classifer_name == 'LR':
		# 	mdl = LogisticRegression(max_iter=1000)
		# 	mdl.fit(x_train, y_train)
		# elif classifer_name == 'KNN':
		# 	mdl = KNeighborsClassifier()
		# 	mdl.fit(x_train, y_train)
		# elif classifer_name == 'SGD':
		# 	mdl = SGDClassifier()
		# 	mdl.fit(x_train, y_train)
		# elif classifer_name == 'RF':
		# 	mdl = RandomForestClassifier()
		# 	mdl.fit(x_train, y_train)
		# else:
		# 	print('miss classifer')
		# 	continue
		
		# F1
		# y_pred = mdl.predict(x_valid)
		# Acc       = np.sum(y_valid == y_pred)  / num_valid
		# print("Accuracy:", 100 * Acc)
		# acc = accuracy_score(y_valid, y_pred)
		# F1_score = f1_score(y_valid, y_pred)
		# print("F1_score:", F1_score)
		bestFitarr = np.squeeze(fmdl['c'])
		F1 = np.max(bestFitarr)
		# number of selected features
		num_feat = fmdl['nf']
		print("Feature Size:", num_feat)
		time_2 = time.time()
		time_cost = time_2 - time_1
		with open('../result/calculation_statistic.txt', 'a') as f:  # 设置文件对象
			f.write(str(time_cost) + ',')  # 将字符串写入文件中
			f.close()
		with open('../result/calculation_F1.txt', 'a') as f:  # 设置文件对象
			f.write(str(F1) + ',')  # 将字符串写入文件中
			f.close()
		# with open('../result/calculation_acc.txt', 'a') as f:  # 设置文件对象
		#	f.write(str(acc) + ',')  # 将字符串写入文件中
		#	f.close()
		with open('../result/calculation_num_Feature.txt', 'a') as f:  # 设置文件对象
			f.write(str(num_feat) + ',')  # 将字符串写入文件中
			f.close()
# # plot convergence
# curve   = fmdl['curve']
# curve   = curve.reshape(np.size(curve,1))
# x       = np.arange(0, opts['T'], 1.0) + 1.0
#
# fig, ax = plt.subplots()
# ax.plot(x, curve, 'o-')
# ax.set_xlabel('Number of Iterations')
# ax.set_ylabel('Fitness')
# ax.set_title('PSO')
# ax.grid()
# plt.show()