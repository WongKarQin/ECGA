# encoding=utf-8

import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from sklearn import datasets
from sklearn import svm

if __name__ == '__main__':
	# print('prepare datasets...')
	# Iris数据集
	# iris=datasets.load_iris()
	# features=iris.data
	# labels=iris.target
	
	# MINST数据集
	raw_data = pd.read_csv('../data/UCI_phishing_web_original.txt', header=None)  # 读取csv数据
	data = raw_data.values
	# print(len(data))#5848
	# print(data[0])#
	# print(len(data[0]))#31
	# features = data[::, 1::]
	# labels = data[::, 0]  # 选取33%数据作为测试集，剩余为训练集
	# print("len faetures {} \t len labels {}".format(len(features),len(labels)))
	# len faetures 42000 	 len labels 42000
	# print(len(features[0]))#784
	# print(labels[0])#1
	# len faetures 42000 	 len labels 42000
	features = data[::, :-1]
	labels = data[::, -1]
	# print("len faetures {} \t len labels {}".format(len(features), len(labels)))
	# 5848#5848
	# print(features[0])
	# print(len(features[0]))
	# print(labels[0:20])
	# print(features[20])
	train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,
	                                                                            random_state=0)
	
	time_2 = time.time()
	# print('Start training...')
	clf = svm.SVC()  # svm class
	clf.fit(train_features, train_labels)  # training the svc model
	time_3 = time.time()
	# print('training cost %f seconds' % (time_3 - time_2))
	
	# print('Start predicting...')
	test_predict = clf.predict(test_features)
	time_4 = time.time()
	# print('predicting cost %f seconds' % (time_4 - time_3))
	
	score = accuracy_score(test_labels, test_predict)
	f1_score = f1_score(test_labels, test_predict)
	print('time cost is %f ' % (time_4 - time_2))
	print("The accruacy score is %f" % score)
	print('F1 score is %f' % f1_score)
	with open('../result/calculation_statistic.txt', 'a') as f:  # 设置文件对象
		f.write(str(time_4 - time_2) + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_F1.txt', 'a') as f:  # 设置文件对象
		f.write(str(f1_score) + ',')  # 将字符串写入文件中
		f.close()