#1 <= FNCNO <=60
##load FncData
##FNCNO = 1 ; set_fnc_settings(FNCNO);  out = algo_besd(fnc,[],30,dim,low,up,10e3);
#10e3=10000
maxcycle = 100    #最大迭代数10e3=10,000(常量)
pop_size = 100       #种群规模为30(常量) 每次迭代生成30棵树
dim = 30            #染色体长度为dim 30个特征数
low = 0       #下界 0代表琼该特征
up = 1        #上界 1代表选用该特征作为结点
rate_init = 0.5     #生成初始种群时候，选择某一特征的概率
rate_mutation = 0.4     #发生变异从0变为1或者从1变为0的概率
rate_CR = 0.8       #交叉概率，小于该概率新个体的某个维度取变异个体，否则取父代