from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
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


raw_data = pd.read_csv('../data/duplicate_Mendeley_no_negative.txt', header=None)
data = raw_data.values
# F1_dict ={"num_feature":
# ,"F1_score":
# ,"classifier":
# }
str_list = ['CART','SVM','MLP','BNB','GNB','MNB','CONB','CANB','LR','KNN','SGD','RF']
features = data[::, :-1]
labels = data[::, -1]
dim=len(features[1])
#print(features.shape)#(5849, 30)
for classifer_name in str_list:
	with open('../result/calculation_statistic.txt', 'a') as f:  # 设置文件对象
		f.write('\n'+'chi-square' + classifer_name + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_F1.txt', 'a') as f:  # 设置文件对象
		f.write('\n'+'chi-square' + classifer_name + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_acc.txt', 'a') as f:  # 设置文件对象
		f.write('\n'+'chi-square' + classifer_name + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_num_Feature.txt', 'a') as f:  # 设置文件对象
		f.write('\n'+'chi-square' + classifer_name + ',')  # 将字符串写入文件中
		f.close()
	for k_value in range(1,dim):
		features_new = SelectKBest(chi2, k=k_value).fit_transform(features, labels)
		#print(type(features_new))#<class 'numpy.ndarray'>
		assert features_new.shape[1]==k_value
		# 随机选取33%数据作为测试集，剩余为训练集
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
			f.write(str(k_value)+',')  # 将字符串写入文件中
			f.close()