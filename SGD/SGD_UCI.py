# encoding=utf-8

import pandas as pd
import time

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from sklearn.linear_model import SGDClassifier

if __name__ == '__main__':
	# print("Start read data...")
	time_1 = time.time()
	
	raw_data = pd.read_csv('../data/UCI_phishing_web_original.txt', header=None)
	data = raw_data.values
	
	features = data[::, :-1]
	labels = data[::, -1]
	
	# 随机选取33%数据作为测试集，剩余为训练集
	train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size=0.33,
	                                                                            random_state=0)
	
	time_2 = time.time()
	# print('read data cost %f seconds' % (time_2 - time_1))
	
	# print('Start training...')
	# multi_class可选‘ovr’, ‘multinomial’，默认为ovr用于二类分类，multinomial用于多类分类
	clf = SGDClassifier()
	clf.fit(train_features, train_labels)
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