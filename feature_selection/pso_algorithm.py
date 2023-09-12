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
#part 2

import numpy as np
from numpy.random import rand
from fitness_function import Fun

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
    t = 0
    
    while t < max_iter:
        # Binary conversion
        Xbin = binary_conversion(X, thres, N, dim)
        
        # Fitness
        cnt_replace = 0
        for i in range(N):
            fit[i, 0] = Fun(xtrain, ytrain, Xbin[i, :], opts)
            if fit[i, 0] > fitP[i, 0]:
                Xpb[i, :] = X[i, :]
                fitP[i, 0] = fit[i, 0]
                cnt_replace+=1
            if fitP[i, 0] > fitG:
                Xgb[0, :] = Xpb[i, :]
                fitG = fitP[i, 0]
        print('更新替换数',cnt_replace)
        # Store result
        curve[0, t] = fitG.copy()
        print("Iteration:", t + 1)
        print("Best (PSO):", curve[0, t])
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
    
    # Best feature subset
    Gbin = binary_conversion(Xgb, thres, 1, dim)
    Gbin = Gbin.reshape(dim)
    pos = np.asarray(range(0, dim))
    sel_index = pos[Gbin == 1]
    num_feat = len(sel_index)
    # Create dictionary
    pso_data = {'selected_features_index': sel_index, 'curve': curve, 'num_features': num_feat}
    print(pso_data)
    return pso_data