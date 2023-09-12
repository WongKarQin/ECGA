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

# error rate
def error_rate(xtrain, ytrain, x, opts):
	# parameters
	fold = opts['fold']
	xt = fold['xt']
	yt = fold['yt']
	xv = fold['xv']
	yv = fold['yv']
	classifer_name = opts['classifer']
	# Number of instances
	num_train = np.size(xt, 0)
	num_valid = np.size(xv, 0)
	# Define selected features
	xtrain = xt[:, x == 1]
	ytrain = yt.reshape(num_train)  # Solve bug
	xvalid = xv[:, x == 1]
	yvalid = yv.reshape(num_valid)  # Solve bug
	# Training
	if classifer_name == 'CART':
		mdl = DecisionTreeClassifier(criterion='gini')
		mdl.fit(xtrain, ytrain)
	elif classifer_name == 'SVM':
		mdl = svm.SVC()
		mdl.fit(xtrain, ytrain)
	elif classifer_name == 'MLP':
		mdl = Perceptron()
		mdl.fit(xtrain, ytrain)
	elif classifer_name == 'BNB':
		mdl = BernoulliNB()  # 采用伯努利贝叶斯
		mdl.fit(xtrain, ytrain)
	elif classifer_name == 'GNB':
		mdl = GaussianNB()  # 采用高斯贝叶斯
		mdl.fit(xtrain, ytrain)
	elif classifer_name == 'MNB':
		mdl = MultinomialNB()  # 采用多项式贝叶斯
		mdl.fit(xtrain, ytrain)
	elif classifer_name == 'CONB':
		mdl = ComplementNB()  # 采用互补朴素贝叶斯
		mdl.fit(xtrain, ytrain)
	elif classifer_name == 'CANB':
		mdl = CategoricalNB()  # 采用绝对贝叶斯
		mdl.fit(xtrain, ytrain)
	elif classifer_name == 'LR':
		mdl = LogisticRegression(max_iter=1000)
		mdl.fit(xtrain, ytrain)
	elif classifer_name == 'KNN':
		mdl = KNeighborsClassifier()
		mdl.fit(xtrain, ytrain)
	elif classifer_name == 'SGD':
		mdl = SGDClassifier()
		mdl.fit(xtrain, ytrain)
	elif classifer_name == 'RF':
		mdl = RandomForestClassifier()
		mdl.fit(xtrain, ytrain)
	else:
		print('miss classifer')
		return 0.1
	# Prediction
	ypred = mdl.predict(xvalid)
	#acc = np.sum(yvalid == ypred) / num_valid
	#acc = accuracy_score(yvalid, ypred)
	#error = 1-acc
	F1_score = f1_score(yvalid, ypred)
	# error = 1 - F1_score
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