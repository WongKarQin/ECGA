"""
使用RFE进行特征选择：RFE是常见的特征选择方法，也叫递归特征消除。它的工作原理是递归删除特征，
并在剩余的特征上构建模型。它使用模型准确率来判断哪些特征（或特征组合）对预测结果贡献较大。
"""

from sklearn import datasets
from sklearn.feature_selection import RFE
import pandas as pd
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn.tree import DecisionTreeClassifier
import sys
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB,ComplementNB,CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR,SVC

raw_data = pd.read_csv('../data/duplicate_Mendeley_no_negative.txt', header=None)
data = raw_data.values
#str_list = ['CART','SVM','MLP','BNB','GNB','MNB','CONB','CANB','LR','KNN','SGD','RF']
str_list = ['CART','KNN']
features = data[::, :-1]
labels = data[::, -1]
dim = len(features[1])
for classifer_name in str_list :
	with open('../result/calculation_statistic.txt', 'a') as f:  # 设置文件对象
		f.write('\n'+'RFE' + classifer_name + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_F1.txt', 'a') as f:  # 设置文件对象
		f.write('\n'+'RFE' + classifer_name + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_acc.txt', 'a') as f:  # 设置文件对象
		f.write('\n'+'RFE' + classifer_name + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_num_Feature.txt', 'a') as f:  # 设置文件对象
		f.write('\n'+'RFE' + classifer_name + ',')  # 将字符串写入文件中
		f.close()
	if classifer_name == 'CART':
		model = DecisionTreeClassifier(criterion='gini')
	elif classifer_name == 'SVM':
		model = svm.SVC()
	elif classifer_name == 'MLP':
		model = Perceptron()
	elif classifer_name == 'BNB':
		model = BernoulliNB()
	elif classifer_name == 'GNB':
		model = GaussianNB()
	elif classifer_name == 'MNB':
		model = MultinomialNB()
	elif classifer_name == 'CONB':
		model = ComplementNB()
	elif classifer_name == 'CANB':
		model = CategoricalNB()
	elif classifer_name =='LR':
		model = LogisticRegression(max_iter=1000)
	elif classifer_name == 'KNN':
		model = KNeighborsClassifier()
	elif classifer_name == 'SGD':
		model = SGDClassifier()
	elif classifer_name == 'RF':
		model = RandomForestClassifier()
	else:
		print('miss classifer')
		sys.exit(1)
	for feature_selection_num in range(1,dim):
		print(feature_selection_num)
		rfe = RFE(estimator = model, n_features_to_select = feature_selection_num )
		#rfe(estimator,n_features_to_select,step) estimator参数指明基模型，n_features_to_select指定最终要保留的特征数量，step为整数时设置每次要删除的特征数量，当小于1时，每次去除权重最小的特征。
		try:
			rfe = rfe.fit(features, labels)
		except RuntimeError:
			model = SVC(kernel="linear")
			rfe = RFE(estimator=model, n_features_to_select=feature_selection_num)
			rfe = rfe.fit(features, labels)
		except AttributeError:
			model = SVC(kernel="linear")
			rfe = RFE(estimator=model, n_features_to_select=feature_selection_num)
			rfe = rfe.fit(features, labels)
		# print(type(rfe))#<class 'sklearn.feature_selection._rfe.RFE'>
		# print(rfe)#RFE(estimator=LogisticRegression(max_iter=1000), n_features_to_select=3)
		# print(rfe.support_)
		# print(rfe.ranking_)
		# print(rfe.n_features_)
		index_feature_selection_list = []
		for index_feature,item_TF in enumerate(rfe.support_):
			if item_TF == True:index_feature_selection_list.append(index_feature)
		#print(index_feature_selection_list)
		features_new = features[::, index_feature_selection_list]
		assert features_new.shape[1] == rfe.n_features_
		train_features, test_features, train_labels, test_labels = train_test_split(features_new, labels, test_size=0.33,
		                                                                            random_state=0)
		time_1 = time.time()
		if classifer_name=='CART':
			clf = DecisionTreeClassifier(criterion='gini')
			clf.fit(train_features, train_labels)
			test_predict = clf.predict(test_features)
		elif classifer_name =='SVM':
			clf = svm.SVC()
			clf.fit(train_features, train_labels)
			test_predict = clf.predict(test_features)
		elif classifer_name =='MLP':
			clf = Perceptron()
			clf.fit(train_features, train_labels)
			test_predict = clf.predict(test_features)
		elif classifer_name =='BNB':
			clf = BernoulliNB()  # 采用伯努利贝叶斯
			clf.fit(train_features, train_labels)
			test_predict = clf.predict(test_features)
		elif classifer_name =='GNB':
			clf = GaussianNB()  # 采用高斯贝叶斯
			clf.fit(train_features, train_labels)
			test_predict = clf.predict(test_features)
		elif classifer_name =='MNB':
			clf = MultinomialNB()  # 采用多项式贝叶斯
			clf.fit(train_features, train_labels)
			test_predict = clf.predict(test_features)
		elif classifer_name =='CONB':
			clf = ComplementNB()  # 采用互补朴素贝叶斯
			clf.fit(train_features, train_labels)
			test_predict = clf.predict(test_features)
		elif classifer_name =='CANB':
			clf = CategoricalNB()  # 采用绝对贝叶斯
			clf.fit(train_features, train_labels)
			test_predict = clf.predict(test_features)
		elif classifer_name =='LR':
			clf = LogisticRegression(max_iter=1000)
			clf.fit(train_features, train_labels)
			test_predict = clf.predict(test_features)
		elif classifer_name =='KNN':
			neigh = KNeighborsClassifier()
			neigh.fit(train_features, train_labels)
			test_predict = neigh.predict(test_features)
		elif classifer_name == 'SGD':
			clf = SGDClassifier()
			clf.fit(train_features, train_labels)
			test_predict = clf.predict(test_features)
		elif classifer_name == 'RF':
			clf = RandomForestClassifier()
			clf.fit(train_features, train_labels)
			test_predict = clf.predict(test_features)
		else:
			print('miss classifer')
			sys.exit(1)
		time_2 = time.time()
		acc = accuracy_score(test_labels, test_predict)
		f1_value = f1_score(test_labels, test_predict)
		with open('../result/calculation_statistic.txt', 'a') as f:  # 设置文件对象
			f.write(str(time_2 - time_1) + ',')  # 将字符串写入文件中
			f.close()
		with open('../result/calculation_F1.txt', 'a') as f:  # 设置文件对象
			f.write(str(f1_value) + ',')  # 将字符串写入文件中
			f.close()
		with open('../result/calculation_acc.txt', 'a') as f:  # 设置文件对象
			f.write(str(acc)+ ',')  # 将字符串写入文件中
			f.close()
		with open('../result/calculation_num_Feature.txt', 'a') as f:  # 设置文件对象
			f.write(str(rfe.n_features_)+',')  # 将字符串写入文件中
			f.close()
'''
[False False False False False  True False False False False False False
 False  True False False False False False False False False False False
 False False False False False False]
[13 26  9 17 10  1  5  2 27 18 19 11 20  1  4  3 21 16  7 23 25 24 22 29
 14  6 28  8 12 15]
'''