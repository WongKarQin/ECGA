# import numpy as np
# import random
#
#
# def fit_fun(X):  # 适应函数
#     return -np.abs(np.sin(X[0]) * np.cos(X[1]) * np.exp(np.abs(1 - np.sqrt(X[0] ** 2 + X[1] ** 2) / np.pi)))
#
#
# class Particle:
#     # 初始化
#     def __init__(self, x_max, max_vel, dim):
#         self.__pos = [random.uniform(-x_max, x_max) for i in range(dim)]  # 粒子的位置
#         self.__vel = [random.uniform(-max_vel, max_vel) for i in range(dim)]  # 粒子的速度
#         self.__bestPos = [0.0 for i in range(dim)]  # 粒子最好的位置
#         self.__fitnessValue = fit_fun(self.__pos)  # 适应度函数值
#
#     def set_pos(self, i, value):
#         self.__pos[i] = value
#
#     def get_pos(self):
#         return self.__pos
#
#     def set_best_pos(self, i, value):
#         self.__bestPos[i] = value
#
#     def get_best_pos(self):
#         return self.__bestPos
#
#     def set_vel(self, i, value):
#         self.__vel[i] = value
#
#     def get_vel(self):
#         return self.__vel
#
#     def set_fitness_value(self, value):
#         self.__fitnessValue = value
#
#     def get_fitness_value(self):
#         return self.__fitnessValue
#
#
# class PSO:
#     def __init__(self, dim, size, iter_num, x_max, max_vel, best_fitness_value=float('Inf'), C1=2, C2=2, W=1):
#         self.C1 = C1
#         self.C2 = C2
#         self.W = W
#         self.dim = dim  # 粒子的维度
#         self.size = size  # 粒子个数
#         self.iter_num = iter_num  # 迭代次数
#         self.x_max = x_max
#         self.max_vel = max_vel  # 粒子最大速度
#         self.best_fitness_value = best_fitness_value
#         self.best_position = [0.0 for i in range(dim)]  # 种群最优位置
#         self.fitness_val_list = []  # 每次迭代最优适应值
#
#         # 对种群进行初始化
#         self.Particle_list = [Particle(self.x_max, self.max_vel, self.dim) for i in range(self.size)]
#
#     def set_bestFitnessValue(self, value):
#         self.best_fitness_value = value
#
#     def get_bestFitnessValue(self):
#         return self.best_fitness_value
#
#     def set_bestPosition(self, i, value):
#         self.best_position[i] = value
#
#     def get_bestPosition(self):
#         return self.best_position
#
#     # 更新速度
#     def update_vel(self, part):
#         for i in range(self.dim):
#             vel_value = self.W * part.get_vel()[i] + self.C1 * random.random() * (part.get_best_pos()[i] - part.get_pos()[i]) \
#                         + self.C2 * random.random() * (self.get_bestPosition()[i] - part.get_pos()[i])
#             if vel_value > self.max_vel:
#                 vel_value = self.max_vel
#             elif vel_value < -self.max_vel:
#                 vel_value = -self.max_vel
#             part.set_vel(i, vel_value)
#
#     # 更新位置
#     def update_pos(self, part):
#         for i in range(self.dim):
#             pos_value = part.get_pos()[i] + part.get_vel()[i]
#             part.set_pos(i, pos_value)
#         value = fit_fun(part.get_pos())
#         if value < part.get_fitness_value():
#             part.set_fitness_value(value)
#             for i in range(self.dim):
#                 part.set_best_pos(i, part.get_pos()[i])
#         if value < self.get_bestFitnessValue():
#             self.set_bestFitnessValue(value)
#             for i in range(self.dim):
#                 self.set_bestPosition(i, part.get_pos()[i])
#
#     def update(self):
#         for i in range(self.iter_num):
#             for part in self.Particle_list:
#                 self.update_vel(part)  # 更新速度
#                 self.update_pos(part)  # 更新位置
#             self.fitness_val_list.append(self.get_bestFitnessValue())  # 每次迭代完把当前的最优适应度存到列表
#         return self.fitness_val_list, self.get_bestPosition()
#
# part 2

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
		abs_energy =  fitness_np_array
		abs_energy = abs_energy.tolist()
	elif problem_type == 'min':
		abs_energy = -1 * fitness_vector
	if t == 20:
		active_window = [np.min(abs_energy), np.max(abs_energy)]
	elif t>20:
		abs_energy_new = abs_energy.copy()
		abs_energy_new.append(record_active_window[0])
		abs_energy_new.append(record_active_window[1])
		active_window = [np.min(abs_energy_new), np.max(abs_energy_new)]
	lt = active_window[0]
	ut = active_window[1]
	#lt = np.array(active_window[0])
	#ut = np.array(active_window[1])
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


def init_velocity(lb, ub, N, dim):
	V = np.zeros([N, dim], dtype='float')
	Vmax = np.zeros([1, dim], dtype='float')
	Vmin = np.zeros([1, dim], dtype='float')
	# Maximum & minimum velocity
	for d in range(dim):
		Vmax[0, d] = (ub[0, d] - lb[0, d]) / 2
		Vmin[0, d] = -Vmax[0, d]
	
	for i in range(N):
		for d in range(dim):
			V[i, d] = Vmin[0, d] + (Vmax[0, d] - Vmin[0, d]) * rand()
	
	return V, Vmax, Vmin


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
	w = 0.9  # inertia weight
	c1 = 2  # acceleration factor
	c2 = 2  # acceleration factor
	
	N = opts['N']
	max_iter = opts['T']
	if 'w' in opts:
		w = opts['w']
	if 'c1' in opts:
		c1 = opts['c1']
	if 'c2' in opts:
		c2 = opts['c2']
	
	# Dimension
	dim = np.size(xtrain, 1)
	if np.size(lb) == 1:
		ub = ub * np.ones([1, dim], dtype='float')
		lb = lb * np.ones([1, dim], dtype='float')
	
	# Initialize position & velocity
	X = init_position(lb, ub, N, dim)
	V, Vmax, Vmin = init_velocity(lb, ub, N, dim)
	
	# Pre
	fit = np.zeros([N, 1], dtype='float')
	Xgb = np.zeros([1, dim], dtype='float')
	fitG = float('-inf')
	Xpb = np.zeros([N, dim], dtype='float')
	fitP = float('-inf') * np.ones([N, 1], dtype='float')
	curve = np.zeros([1, max_iter], dtype='float')
	meanFit = np.zeros([1, max_iter], dtype='float')
	with open('../result/fitness_each_generation', 'a') as f:  # 设置文件对象
		f.write('\n'+'ECPSO'+',')  # 将字符串写入文件中
		f.close()
	t = 0
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
		if t > 0.6 * max_iter and rating_based_entropy_value < 0.05 :
			print("算法已经收敛，程序结束")
			flag_end = True
			break
		# Binary conversion
		Xbin = binary_conversion(X, thres, N, dim)
		if flag_new == True:
			flag_new = False
			for individual in Xbin:
				if len(feature_share_abandoned_list) == 0 and len(feature_share_choosed_list) == 0:
					break
				if len(feature_share_choosed_list) != 0:
					for item_choosed in feature_share_choosed_list:
						individual[item_choosed] = 1
				if len(feature_share_abandoned_list) != 0:
					for item_abandoned in feature_share_abandoned_list:
						individual[item_abandoned] = 0
		# Fitness
		cnt_replace = 0
		for i in range(N):
			fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
			if fit[i, 0] > fitP[i, 0]:
				Xpb[i, :] = X[i, :]
				fitP[i, 0] = fit[i, 0]
				cnt_replace += 1
			if fitP[i, 0] > fitG:
				Xgb[0, :] = Xpb[i, :]
				fitG = fitP[i, 0]
				cnt_best = 0
		cnt_best +=1
		#print('子代替换父代数{}\t最优停滞轮数{}'.format(cnt_replace,cnt_best))
		
		# Store result
		curve[0, t] = fitG.copy()
		meanFit[0, t] = np.mean(fit)
		with open('../result/fitness_each_generation', 'a') as f:  # 设置文件对象
			f.write(str(meanFit[0, t]) + ',')  # 将字符串写入文件中
			f.close()
		print("Iteration:", t + 1,"Best (PSO):", curve[0, t],'子代替换父代数',cnt_replace,'最优停滞轮数',cnt_best)
		#print("Best (PSO):", curve[0, t])
		t += 1
		
		for i in range(N):
			for d in range(dim):
				# Update velocity
				r1 = rand()
				r2 = rand()
				V[i, d] = w * V[i, d] + c1 * r1 * (Xpb[i, d] - X[i, d]) + c2 * r2 * (Xgb[0, d] - X[i, d])
				# Boundary
				V[i, d] = boundary(V[i, d], Vmin[0, d], Vmax[0, d])
				# Update position
				X[i, d] = X[i, d] + V[i, d]
				# Boundary
				X[i, d] = boundary(X[i, d], lb[0, d], ub[0, d])
		
		
		if t > 30 and rand()>0.4 and rating_based_entropy_value>0.05:
			cnt_start_new +=1
			print("新机制介入，连续{}轮迭代无法寻优".format(cnt_best))
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
					a=0.7, b=0.3, part_num=3)
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
	pso_data = {'selected_features_index': sel_index, 'curve': curve, 'num_features': num_feat}
	print(pso_data)
	print('新机制使用次数', cnt_start_new)
	return pso_data