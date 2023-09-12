# [1997]-"Differential evolution - A simple and efficient heuristic for global optimization over continuous spaces"

import numpy as np
from numpy.random import rand
from fitness_function import Fun

def rating_information_entropy_calculation(fitness_vector,problem_type,t,record_active_window):
	fitness_vector = np.squeeze(fitness_vector)
	if t == 20:
		record_active_window = []
	else:
		record_active_window = record_active_window
	if problem_type == 'max':
		fitness_np_array = np.array(fitness_vector)
		abs_energy = fitness_np_array
		#abs_energy = abs_energy.tolist()
	elif problem_type == 'min':
		abs_energy = -1 * fitness_vector
	if t == 20:
		active_window = [np.min(abs_energy), np.max(abs_energy)]
	elif t>20:
		abs_energy = abs_energy.tolist()
		abs_energy_new = abs_energy.copy()
		abs_energy_new.append(record_active_window[0])
		abs_energy_new.append(record_active_window[1])
		active_window = [np.min(abs_energy_new), np.max(abs_energy_new)]
	#lt = float(str(active_window[0]).strip('[')[:-1])
	lt = active_window[0]
	ut = active_window[1]
	#print(type(active_window[0]), active_window[0])
	#print(type(active_window), active_window)
	#lt = np.array(active_window[0])
	#ut = active_window[1]
	#print('lt{}\tut{}'.format(lt,ut))
	#print(ut - lt)
	K = t+2 #上中下
	a = 1.2
	grade_list = []
	for i in range(0, K):  # i为第i个等级数，取整,0到K-1
		constant_1 = np.power(a, i - 1) - 1
		constant_2 = np.power(a, K - 1) - 1
		constant_3 = np.power(a, i) - 1
		grade_1 = constant_1 / constant_2 * (ut - lt) + lt
		grade_2 = constant_3 / constant_2 * (ut - lt) + lt
		grade = [max(grade_1, lt), min(grade_2, ut)]
		# print('GRADE',grade)
		if grade[0] != grade[1]:
			grade_list.append(grade)
	#为了防止绝对能量有溢出grade_list的情况，加入保险等级
	min_num = grade_list[0][0]
	max_num = grade_list[-1][1]
	if min_num!=lt:
		grade_list.insert(0,[lt,min_num])
	if max_num!=ut:
		grade_list.append([max_num,ut])
	N = len(abs_energy)
	K = len(grade_list)
	rating_based_entropy_value = 0
	n_grade_list = [0] * K  # 每个等级所拥有的个体数[1,1,2,4]共4个等级，共8个个体
	grade_index_list_dict = {}  # 种群个体索引号与所处的等级
	for index_population, abs_energy_value in enumerate(abs_energy):
		for grade_index, grade_value in enumerate(grade_list):
			# n = 0
			if abs_energy_value >= grade_value[0] and abs_energy_value <=grade_value[1]:
				# print(round(abs_energy_value, self.t + 1), round(grade_value[0], self.t + 1),round(grade_value[1], self.t + 1))
				# n += 1
				grade_index_list_dict[index_population] = grade_index
				n_grade_list[grade_index] = n_grade_list[grade_index] + 1
				break
	#print(grade_index_list_dict.keys())
	try:
		assert sum(n_grade_list) == N
	except AssertionError:
		print('划分落入等级个体总数不等于N\tn_grade_list:{}\t累计求和:{}\tN:{}\tgrade_list{}'.format(
			n_grade_list, sum(n_grade_list), N,grade_list))
		print(len(grade_index_list_dict.keys()))
		for i in range(100):
			if i not in grade_index_list_dict.keys():
				print('i', i, '绝对能量', abs_energy[i])
		n_grade_list[0] = 1
		print('after', n_grade_list)
	# 问题越到后面，划分越精细，请问，个体如何落入等级。编程比较困难。1个个体有且仅有1个等级。
	assert sum(n_grade_list) == N
	# print('1',n_grade_list)
	for n_value in n_grade_list:
		if n_value == 0: continue
		rating_based_entropy_value += -1 * n_value / N * np.log(n_value / N) / np.log(K)
	try:
		assert rating_based_entropy_value >= 0 and rating_based_entropy_value <= 1
	except AssertionError:
		print('等级熵计算结果范围不在0到1', rating_based_entropy_value)
	return rating_based_entropy_value,active_window,n_grade_list, grade_list,K

def new_mechanism(X,fit,a,b,part_num):
	dim = len(X[1])
	fitness_index_sorted = np.argsort(fit)
	top_num = 100 // part_num
	index_choosed = fitness_index_sorted[-top_num:]
	# print(index_choosed)
	pop_copy = np.copy(X)
	pop_copy_np_array = np.array(pop_copy)
	pop_new_flag = np.copy(pop_copy_np_array[index_choosed])
	#print('pop_new_flag',pop_new_flag)
	feature_share_list = [0] * dim
	feature_share_choosed_list = []
	feature_share_abandoned_list = []
	feature_share_random_list = []
	for top_individual in pop_new_flag:
		#print('top_individual',top_individual)
		for feature_index_1, feature_selection_value_1 in enumerate(top_individual):
			#print(feature_selection_value)
			for feature_index_2, feature_selection_value_2 in enumerate(feature_selection_value_1):
				if feature_selection_value_2 == 1: feature_share_list[feature_index_2] += 1
	# 二b八a原理，幂律分布
	for index, item in enumerate(feature_share_list):
		if item >= a * top_num:
			feature_share_choosed_list.append(index)
		elif item <= b * top_num:
			feature_share_abandoned_list.append(index)
		else:
			feature_share_random_list.append(index)
	return feature_share_choosed_list,feature_share_abandoned_list,feature_share_random_list

def init_position(lb, ub, N, dim):
	X = np.zeros([N, dim], dtype='float')
	for i in range(N):
		for d in range(dim):
			X[i, d] = lb[0, d] + (ub[0, d] - lb[0, d]) * rand()
	
	return X


def binary_conversion(X, thres, N, dim):
	Xbin = np.zeros([N, dim], dtype='int')
	for i in range(N):
		for d in range(dim):
			if X[i, d] > thres:
				Xbin[i, d] = 1
			else:
				Xbin[i, d] = 0
	
	return Xbin


def boundary(x, lb, ub):
	if x < lb:
		x = lb
	if x > ub:
		x = ub
	
	return x


def jfs(xtrain, ytrain, opts):
	# Parameters
	ub = 1
	lb = 0
	thres = 0.5
	CR = 0.9  # crossover rate
	F = 0.5  # factor
	
	N = opts['N']
	max_iter = opts['T']
	if 'CR' in opts:
		CR = opts['CR']
	if 'F' in opts:
		F = opts['F']
	
	# Dimension
	dim = np.size(xtrain, 1)
	if np.size(lb) == 1:
		ub = ub * np.ones([1, dim], dtype='float')
		lb = lb * np.ones([1, dim], dtype='float')
	
	# Initialize position
	X = init_position(lb, ub, N, dim)
	# Binary conversion
	Xbin = binary_conversion(X, thres, N, dim)
	
	# Fitness at first iteration
	fit = np.zeros([N, 1], dtype='float')
	Xgb = np.zeros([1, dim], dtype='float')
	#fitG = float('inf')
	fitG = float('-inf')
	for i in range(N):
		fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
		if fit[i, 0] > fitG:
			Xgb[0, :] = X[i, :]
			fitG = fit[i, 0]
	
	# Pre
	curve = np.zeros([1, max_iter], dtype='float')
	meanFit = np.zeros([1, max_iter], dtype='float')
	t = 0
	with open('../result/fitness_each_generation', 'a') as f:  # 设置文件对象
		f.write('\n'+'ECDE'+',')  # 将字符串写入文件中
		f.close()
	curve[0, t] = fitG.copy()
	meanFit[0, t] = np.mean(fit)
	with open('../result/fitness_each_generation', 'a') as f:  # 设置文件对象
		f.write(str(meanFit[0, t]) + ',')  # 将字符串写入文件中
		f.close()
	print("Generation:", t + 1,"Best (DE):", curve[0, t])
	t += 1
	cnt_best = 0
	flag_end = False
	flag_new = False
	feature_share_choosed_list = []
	feature_share_abandoned_list = []
	cnt_start_new = 0
	
	while t < max_iter:
		if t == 20:
			rating_based_entropy_value, active_window, n_grade_list, grade_list, K = rating_information_entropy_calculation(
				fitness_vector=fit, problem_type='max', t=t, record_active_window=[])
			print('种群整体等级熵{}\t适应值上下界活动窗口{}\t落入各个等级个体数{}\t{}个等级'.format(rating_based_entropy_value, active_window,
			                                                           n_grade_list, K))
		elif t > 20:
			record_window = active_window
			rating_based_entropy_value, active_window, n_grade_list, grade_list, K = rating_information_entropy_calculation(
				fitness_vector=fit, problem_type='max', t=t, record_active_window=record_window)
			print('种群整体等级熵{}\t适应值上下界活动窗口{}\t落入各个等级个体数{}\t{}个等级'.format(rating_based_entropy_value, active_window,
			                                                           n_grade_list, K))
		if t > 20 and rating_based_entropy_value < 0.05 and t > 0.6 * max_iter:
			print("算法已经收敛，程序结束")
			flag_end = True
			break
		V = np.zeros([N, dim], dtype='float')
		U = np.zeros([N, dim], dtype='float')
		
		for i in range(N):
			# Choose r1, r2, r3 randomly, but not equal to i
			RN = np.random.permutation(N)
			for j in range(N):
				if RN[j] == i:
					RN = np.delete(RN, j)
					break
			
			r1 = RN[0]
			r2 = RN[1]
			r3 = RN[2]
			# mutation (2)
			for d in range(dim):
				V[i, d] = X[r1, d] + F * (X[r2, d] - X[r3, d])
				# Boundary
				V[i, d] = boundary(V[i, d], lb[0, d], ub[0, d])
			
			# Random one dimension from 1 to dim
			index = np.random.randint(low=0, high=dim)
			# crossover (3-4)
			for d in range(dim):
				if (rand() <= CR) or (d == index):
					U[i, d] = V[i, d]
				else:
					U[i, d] = X[i, d]
		
		# Binary conversion
		Ubin = binary_conversion(U, thres, N, dim)
		if flag_new == True:
			flag_new = False
			for individual in Ubin:
				if len(feature_share_abandoned_list) == 0 and len(feature_share_choosed_list) == 0:
					break
				if len(feature_share_choosed_list) != 0:
					for item_choosed in feature_share_choosed_list:
						individual[item_choosed] = 1
				if len(feature_share_abandoned_list) != 0:
					for item_abandoned in feature_share_abandoned_list:
						individual[item_abandoned] = 0
		
		cnt_replace = 0
		# Selection
		for i in range(N):
			fitU = Fun(xtrain, ytrain, Ubin[i, :], opts)
			if fitU >= fit[i, 0]:
				X[i, :] = U[i, :]
				fit[i, 0] = fitU
				
				cnt_replace += 1
			if fit[i, 0] > fitG:
				Xgb[0, :] = X[i, :]
				fitG = fit[i, 0]
				cnt_best = 0
		cnt_best += 1
		Xbin = binary_conversion(X, thres, N, dim)
		# print('本轮迭代寻优替换数', cnt_replace)
		# Store result
		curve[0, t] = fitG.copy()
		meanFit[0,t] = np.mean(fit)
		with open('../result/fitness_each_generation', 'a') as f:  # 设置文件对象
			f.write(str(meanFit[0,t]) + ',')  # 将字符串写入文件中
			f.close()
		print("Generation:", t + 1, "Best (DE):", curve[0, t], '本轮迭代寻优替换数', cnt_replace)
		# print("Best (DE):", curve[0, t])
		t += 1
		if t > 30 and rand()>0.4 and rating_based_entropy_value>0.05:
			cnt_start_new +=1
			print("新机制介入，连续{}轮迭代无法寻优".format(cnt_best),"共享机制调用次数：",cnt_start_new)
			flag_new = True
			feature_share_choosed_list, feature_share_abandoned_list, feature_share_random_list = new_mechanism(Xbin,fit,
				a=0.9, b=0.1, part_num=3)
			print('choosed', feature_share_choosed_list, len(feature_share_choosed_list))
			print('abandoned', feature_share_abandoned_list, len(feature_share_abandoned_list))
			print('random', feature_share_random_list, len(feature_share_random_list))
			if feature_share_choosed_list == []:
				feature_share_choosed_list, feature_share_abandoned_list, feature_share_random_list = new_mechanism(Xbin,fit,
					a=0.7, b=0.3, part_num=3)
				print('2 choosed', feature_share_choosed_list, len(feature_share_choosed_list))
				print('1 abandoned', feature_share_abandoned_list, len(feature_share_abandoned_list))
				print('2 random', feature_share_random_list, len(feature_share_random_list))
			if feature_share_abandoned_list == []:
				feature_share_choosed_list, feature_share_abandoned_list, feature_share_random_list = new_mechanism(Xbin,fit,
					a=0.9, b=0.1, part_num=3)
				print('1 choosed', feature_share_choosed_list, len(feature_share_choosed_list))
				print('2 abandoned', feature_share_abandoned_list, len(feature_share_abandoned_list))
				print('2 random', feature_share_random_list, len(feature_share_random_list))
			if i > 0.6 * max_iter and rating_based_entropy_value > 0.3 and len(
				feature_share_random_list) > dim // 4:
				feature_share_choosed_list, feature_share_abandoned_list, feature_share_random_list = new_mechanism(Xbin,fit,
					a=0.7, b=0.3, part_num=2)
				print('sprint choosed', feature_share_choosed_list, len(feature_share_choosed_list))
				print('sprint abandoned', feature_share_abandoned_list, len(feature_share_abandoned_list))
				print('sprint random', feature_share_random_list, len(feature_share_random_list))
	# Best feature subset
	Gbin = binary_conversion(Xgb, thres, 1, dim)
	Gbin = Gbin.reshape(dim)
	pos = np.asarray(range(0, dim))
	sel_index = pos[Gbin == 1]
	num_feat = len(sel_index)
	# Create dictionary
	de_data = {'selected_features_index': sel_index, 'curve': curve, 'num_features': num_feat}
	print(de_data)
	return de_data