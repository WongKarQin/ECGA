# encoding=utf-8

import pandas as pd
import numpy as np
import time

from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

if __name__ == '__main__':
	# print("Start read data...")
	
	raw_data = pd.read_csv('../data/UCI_phishing_web_original.txt', header=None)  # 读取csv数据
	data = raw_data.values
	# for data_item in data:
	# 	for index,value in enumerate(data_item):
	# 		if value < 0:
	# 			data_item[index] =2
	
	features = data[::, :-1]
	labels = data[::, -1]
	# print(set(labels))
	# print(features[0:10])
	
	# assert 1==2
	# 随机选取33%数据作为测试集，剩余为训练集
	train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,
	                                                                            random_state=0)
	
	# print('read data cost %f seconds' % (time_2 - time_1))
	
	# print('Start training...')
	time_1 = time.time()
	# clf = MultinomialNB(alpha=1.0)  # 加入laplace平滑
	# clf = MultinomialNB()
	clf = BernoulliNB()  # 采用伯努利贝叶斯较适用于该钓鱼网站分类数据集
	clf.fit(train_features, train_labels)
	time_3 = time.time()
	# print('training cost %f seconds' % (time_3 - time_2))
	
	# print('Start predicting...')
	test_predict = clf.predict(test_features)
	time_4 = time.time()
	# print('predicting cost %f seconds' % (time_4 - time_3))
	
	score = accuracy_score(test_labels, test_predict)
	f1_score = f1_score(test_labels, test_predict)
	print('time cost is %f ' % (time_4 - time_1))
	print("The accruacy score is %f" % score)
	print('F1 score is %f' % f1_score)
	with open('../result/calculation_statistic.txt', 'a') as f:  # 设置文件对象
		f.write(str(time_4 - time_1) + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_F1.txt', 'a') as f:  # 设置文件对象
		f.write(str(f1_score) + ',')  # 将字符串写入文件中
		f.close()