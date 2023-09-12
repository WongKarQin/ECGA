import numpy as np
from scipy.stats import pearsonr
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
features = data[::, :-1]
#print(type(features))#<class 'numpy.ndarray'>
labels = data[::, -1]
dim=len(features[1])
person_correlation_coefficient_of_features = []
feature_index_sort_person = []
person_correlation_coefficient_of_features_dict ={}
# print(features)
# print(type(features))
# print(features[:,0])
# print(features[:,1])
# print(features[:,-1])
# np.random.seed(0)
# size = 300
# x = np.random.normal(0,1,size)
# print(x)
# y_1 = x + np.random.normal(0, 1, size)
# y_2 = x + np.random.normal(0, 10, size)
# print("shape {} {} {}".format(x.shape,y_1.shape,y_2.shape))
# print("value max {} {} {} min {} {} {}".format(np.max(x),np.max(y_1),np.max(y_2)
#                                                ,np.min(x),np.min(y_1),np.min(y_2)))
# # pearsonr(x, y)的输入为特征矩阵和目标向量
# print("Lower noise", pearsonr(x, y_1))
# print("Higher noise", pearsonr(x, y_2))
# # 输出为二元组(sorce, p-value)的数组
# ## 结果 Lower noise (0.71824836862138386, 7.3240173129992273e-49)
# ## 结果 Higher noise (0.057964292079338148, 0.31700993885324746)
for num_row in range(0,dim):
	feature_column = features[:,num_row]
	person_correlation_coefficient = pearsonr(feature_column, labels)[0]
	person_correlation_coefficient_of_features_dict[num_row]=person_correlation_coefficient
new_list_after_person = sorted(person_correlation_coefficient_of_features_dict.items(), key = lambda kv:(np.abs(kv[1]), np.abs(kv[0])),reverse=True)
print(new_list_after_person)
# with open('K_person_sort.txt', 'a') as f:  # 设置文件对象
# 	f.write('\n'+str(new_dict))  # 将字符串写入文件中
# 	f.close()
#print(type(new_list_after_person))#<class 'list'>
list_index_after_sort_feature = []
for item in new_list_after_person:
	list_index_after_sort_feature.append(item[0])
#print(list_index_after_sort_feature)

str_list = ['CART','SVM','MLP','BNB','GNB','MNB','CONB','CANB','LR','KNN','SGD','RF']
#print(features.shape)#(5849, 30)
for classifer_name in str_list :
	with open('../result/calculation_statistic.txt', 'a') as f:  # 设置文件对象
		f.write('\n'+'K_person' + classifer_name + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_F1.txt', 'a') as f:  # 设置文件对象
		f.write('\n'+'K_person' + classifer_name + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_acc.txt', 'a') as f:  # 设置文件对象
		f.write('\n'+'K_person' + classifer_name + ',')  # 将字符串写入文件中
		f.close()
	with open('../result/calculation_num_Feature.txt', 'a') as f:  # 设置文件对象
		f.write('\n'+'K_person' + classifer_name + ',')  # 将字符串写入文件中
		f.close()
	for k_value in range(1,dim):
		feature_selection_index = list_index_after_sort_feature[:k_value]
		#for index_item in feature_selection_index:
		features_new = features[::,feature_selection_index]
		#print(features_new.shape)
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