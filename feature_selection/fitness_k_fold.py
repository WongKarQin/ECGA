import numpy as np
from sklearn.tree import DecisionTreeClassifier
import sys
from sklearn import svm
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB,ComplementNB,CategoricalNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

# error rate
def error_rate(features, labels, x, opts):
	
	# # parameters
	# fold = opts['fold']
	# xt = fold['xt']
	# yt = fold['yt']
	# xv = fold['xv']
	# yv = fold['yv']
	# # Number of instances
	# num_train = np.size(xt, 0)
	# num_valid = np.size(xv, 0)
	# xtrain = xt[:, x == 1]
	# ytrain = yt.reshape(num_train)
	# xvalid = xv[:, x == 1]
	# yvalid = yv.reshape(num_valid)  # Solve bug
	classifer_name = opts['classifer']
	num_train = np.size(features, 0)
	X = features[:, x == 1]
	#y = labels.reshape(num_train)  # Solve bug
	# #K fold 1
	# # Define selected features
	# X = features[:, x == 1]
	# y = labels.reshape(num_train)  # Solve bug
	# K fold 2
	# Training
	data_num = np.arange(0, num_train, 1)
	kf = KFold(n_splits=10,shuffle=True)
	d = kf.split(data_num)
	scores = []
	for train_idx, test_idx in d:
		xtrain = X[train_idx]
		ytrain = labels[train_idx]
		xtest = X[test_idx]
		ytest = labels[test_idx]
		if classifer_name == 'CART':
			mdl = DecisionTreeClassifier(criterion='gini')
			mdl.fit(xtrain, ytrain)
		elif classifer_name == 'KNN':
			mdl = KNeighborsClassifier()
			mdl.fit(xtrain, ytrain)
		else:
			print("miss classifer")
			sys.exit('miss classifer')
		ypred = mdl.predict(xtest)
		F1_score = f1_score(ytest, ypred)
		if F1_score==0:continue
		else:
			scores.append(F1_score)
	# if classifer_name == 'CART':
	# 	mdl = DecisionTreeClassifier(criterion='gini')
	# 	#mdl.fit(xtrain, ytrain)
	# elif classifer_name == 'SVM':
	# 	mdl = svm.SVC()
	# 	mdl.fit(xtrain, ytrain)
	# elif classifer_name == 'MLP':
	# 	mdl = Perceptron()
	# 	mdl.fit(xtrain, ytrain)
	# elif classifer_name == 'BNB':
	# 	mdl = BernoulliNB()  # 采用伯努利贝叶斯
	# 	mdl.fit(xtrain, ytrain)
	# elif classifer_name == 'GNB':
	# 	mdl = GaussianNB()  # 采用高斯贝叶斯
	# 	mdl.fit(xtrain, ytrain)
	# elif classifer_name == 'MNB':
	# 	mdl = MultinomialNB()  # 采用多项式贝叶斯
	# 	mdl.fit(xtrain, ytrain)
	# elif classifer_name == 'CONB':
	# 	mdl = ComplementNB()  # 采用互补朴素贝叶斯
	# 	mdl.fit(xtrain, ytrain)
	# elif classifer_name == 'CANB':
	# 	mdl = CategoricalNB()  # 采用绝对贝叶斯
	# 	mdl.fit(xtrain, ytrain)
	# elif classifer_name == 'LR':
	# 	mdl = LogisticRegression(max_iter=1000)
	# 	mdl.fit(xtrain, ytrain)
	# elif classifer_name == 'KNN':
	# 	mdl = KNeighborsClassifier()
	# 	#mdl.fit(xtrain, ytrain)
	# elif classifer_name == 'SGD':
	# 	mdl = SGDClassifier()
	# 	mdl.fit(xtrain, ytrain)
	# elif classifer_name == 'RF':
	# 	mdl = RandomForestClassifier()
	# 	mdl.fit(xtrain, ytrain)
	# else:
	# 	print('miss classifer')
	# 	return 0.1
	# Prediction
	#ypred = mdl.predict(xvalid)
	#acc = np.sum(yvalid == ypred) / num_valid
	#acc = accuracy_score(yvalid, ypred)
	#error = 1-acc
	#F1_score = f1_score(yvalid, ypred)
	# error = 1 - F1_score
	#scores = cross_val_score(mdl, X, y, cv=10, scoring='f1')
	#F1_score = np.mean(scores)
	F1_score = np.mean(scores)
	return F1_score


# Error rate & Feature size
def Fun(xtrain, ytrain, x, opts):
	# Parameters
	alpha = 0.9
	beta = 1 - alpha
	# Original feature size
	max_feat = len(x)
	# Number of selected features
	num_feat = np.sum(x == 1)
	if num_feat ==0:
		return 0
	# Solve if no feature selected
	if num_feat == 0:
		cost = 0
	else:
		# Get error rate
		F1_score = error_rate(xtrain, ytrain, x, opts)
		# Objective function
		#cost = alpha * error + beta * (max_feat - num_feat )/ max_feat
		#print(type(cost),cost)
		cost = F1_score
	return cost