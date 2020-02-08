# ==================================================
# original author: geatpy
# https://github.com/geatpy-dev/geatpy
# ==================================================
# modified by luojiajie
# ==================================================

import os
import csv
import numpy as np
import geatpy as ea
import matplotlib.pyplot as plt


class MyProblem(ea.Problem): # used in optimization.optimize() , see demo of geatpy for detail meaning
    def __init__(self, M, maxormins, Dim, varTypes, lb, ub, lbin, ubin, f_list):
        self.f_list = f_list # a list of surrogate model object or benchmark function
        ea.Problem.__init__(self, 'MyProblem', M, maxormins, Dim, varTypes, lb, ub, lbin, ubin)


    def aimFunc(self, pop): # objective function
        X = pop.Phen.copy()

        # stack calculated objectives to pop.ObjV (numpy array)
        ObjV_empty_flag = 1 # whether pop.ObjV is empty flag (we use this because np.hstack cannot stack empty array)
        for i in range(len(self.f_list)):
            y = self.f_list[i].calculate(X)
            if ObjV_empty_flag: # if pop.ObjV is empty, it will be directly assigned as the first objective
                pop.ObjV = y
                ObjV_empty_flag = 0 # now pop.ObjV is not empty
            else: # if pop.ObjV is not empty, the next objective will be stacked to it
                pop.ObjV = np.hstack((pop.ObjV, y))


        # pop.CV = np.hstack([2 - x1 - x2, x1 + x2 - 6]) # constraint templet



class optimization():
    def __init__(self,lower_bound,upper_bound,f_list,max_gen=50,pop_size=10,initial_Phen=None,M=None):
        self.lb = lower_bound # lower boundary of variable
        self.ub = upper_bound # upper boundary of variable
        self.f_list = f_list # a list of surrogate model object or benchmark function
        self.MAXGEN = max_gen # max generation
        self.NIND = pop_size # population size

        # default parameters
        self.M = M if M is not None else len(f_list) # number of objectives, which is equal to the number of input surrogate models or benchmark function by default. or you can defined by yourself
        self.maxormins = [1] * self.M # maximize or minimize objective, 1 means minimize, -1 means maximize
        self.Dim = len(lower_bound) # dimension of variable
        self.varTypes = [0] * self.Dim # type of variable, 0 means continuous , 1 means discrete
        self.lbin = [1] * self.Dim # 0 means not include variable lower boundary, 1 means include variable lower boundary
        self.ubin = [1] * self.Dim # 0 means not include variable upper boundary, 1 means include variable upper boundary


        self.initial_Phen=initial_Phen # initial samples for optimization


    def read_csv_to_np(self,file_name):
        """
        read data from .csv and convert to numpy.array
        note: if you use np.loadtxt(), when the data have only one row or one column, the loaded data is not in the desired shape
        """
        csv_reader = csv.reader(open(file_name))
        csv_data = []
        for row in csv_reader:
            csv_data.append(row)

        row_num = len(csv_data)
        col_num = len(csv_data[0])

        data = np.zeros((row_num, col_num))
        for i in range(row_num):
            data[i, :] = np.array(list(map(float, csv_data[i])))
        return data


    def optimize(self):
        """===============================实例化问题对象==========================="""
        problem = MyProblem(self.M, self.maxormins, self.Dim, self.varTypes, self.lb, self.ub, self.lbin, self.ubin, self.f_list)  # 生成问题对象
        """=================================种群设置==============================="""
        Encoding = 'BG'  # 编码方式,使用二进制/格雷编码
        Field = ea.crtfld(Encoding, problem.varTypes, problem.ranges, problem.borders)  # 创建区域描述器
        population = ea.Population(Encoding, Field, self.NIND)  # 实例化种群对象（此时种群还没被初始化，仅仅是完成种群对象的实例化）
        """===============================算法参数设置============================="""
        if self.M==1:
            myAlgorithm = ea.soea_SGA_templet(problem, population)  # 实例化一个算法模板对象, single objective
        else:
            myAlgorithm = ea.moea_NSGA3_templet(problem, population)  # 实例化一个算法模板对象, multi objective

        myAlgorithm.MAXGEN = self.MAXGEN  # 最大进化代数
        myAlgorithm.recOper.XOVR = 0.7  # 交叉概率
        myAlgorithm.mutOper.Pm = 1  # 变异概率
        myAlgorithm.drawing = 0  # 设置绘图方式（0：不绘图；1：绘制结果图；2：绘制目标空间过程动画；3：绘制决策空间过程动画）
        """==========================调用算法模板进行种群进化======================="""
        if self.M == 1: # single objective
            [population, obj_trace, var_trace] = myAlgorithm.run()  # 执行算法模板
            self.obj_trace=obj_trace # save obj_trace for plot
            population.save()  # 把最后一代种群的信息保存到文件中
            # 输出结果
            best_gen = np.argmin(problem.maxormins * obj_trace[:, 1])  # 记录最优种群个体是在哪一代
            best_ObjV = np.array([obj_trace[best_gen, 1]])
            best_var = np.array([var_trace[best_gen]])

            if os.path.exists('best_ObjV.csv') == False:
                np.savetxt('best_ObjV.csv', best_ObjV, delimiter=',')
                np.savetxt('best_var.csv', best_var, delimiter=',')
            else:
                best_ObjV_old=self.read_csv_to_np('best_ObjV.csv')
                best_var_old=self.read_csv_to_np('best_var.csv')
                best_ObjV = np.vstack((best_ObjV_old,best_ObjV))
                best_var = np.vstack((best_var_old, best_var))
                np.savetxt('best_ObjV.csv', best_ObjV, delimiter=',')
                np.savetxt('best_var.csv', best_var, delimiter=',')

            # print('最优的目标函数值为：%s' % (best_ObjV))
            # print('最优的控制变量值为：')
            # for i in range(var_trace.shape[1]):
            #     print(var_trace[best_gen, i])
            # print('有效进化代数：%s' % (obj_trace.shape[0]))
            # print('最优的一代是第 %s 代' % (best_gen + 1))
            # print('评价次数：%s' % (myAlgorithm.evalsNum))
            # print('时间已过 %s 秒' % (myAlgorithm.passTime))
            return population
        else: # multi objective
            NDSet = myAlgorithm.run()  # 执行算法模板，得到帕累托最优解集NDSet
            NDSet.save()  # 把结果保存到文件中
            # 输出
            # print('用时：%s 秒' % (myAlgorithm.passTime))
            # print('非支配个体数：%s 个' % (NDSet.sizes))
            # print('单位时间找到帕累托前沿点个数：%s 个' % (int(NDSet.sizes // myAlgorithm.passTime)))
            return NDSet


    def plot(self,save_ind=0):
        if self.M == 1:  # single objective
            plt.figure(figsize=(10, 8))
            plt.plot(range(1,self.MAXGEN+1),self.obj_trace[:,1],c='black')
            plt.tick_params(labelsize=20)  # axis number size
            font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
            plt.xlabel('Number of generation', font)
            plt.ylabel('best objective value of the population', font)
            # plt.show()
            plt.savefig('optimization_' + str(save_ind) + '.eps')
            plt.close()

        else:  # multi objective
            pass # to be continued




# e.g.
if __name__ == '__main__':
    # ==================================================
    rootpath = r'C:\Users\tomjj\Desktop\demo'  # your saopy file path
    import sys
    sys.path.append(rootpath)  # you can directly import the modules in this folder
    sys.path.append(rootpath + r'\saopy\surrogate_model')
    sys.path.append(rootpath + r'\saopy')
    # ==================================================
    from saopy.function_evaluation.benchmark_func import *
    from saopy.surrogate_model.ANN import *
    from saopy.surrogate_model.KRG import *
    from saopy.surrogate_model.RBF import *
    from saopy.surrogate_model.surrogate_model import *


    flag=0 # 0:single object demo, 1:multi object demo

    if flag==0: # single object demo
        dimension=2

        lower_bound = [-32.768]*dimension # ackley
        upper_bound = [32.768]*dimension

        # lower_bound = [-5.12]*dimension # sphere
        # upper_bound = [5.12]*dimension

        # lower_bound = [-2*3.14159]*dimension # easom
        # upper_bound = [2*3.14159]*dimension

        # lower_bound = [-600]*dimension # griewank
        # upper_bound = [600]*dimension

        # lower_bound = [0]*dimension # michaelwicz
        # upper_bound = [3.14159]*dimension

        # lower_bound = [-1]*dimension # perm
        # upper_bound = [1]*dimension

        # lower_bound = [-5.12]*dimension # rastrigin
        # upper_bound = [5.12]*dimension

        # lower_bound = [-5]*dimension # rosenbrock
        # upper_bound = [5]*dimension

        # lower_bound = [-500]*dimension # schwefel
        # upper_bound = [500]*dimension

        # lower_bound = [-10]*dimension # shubert
        # upper_bound = [10]*dimension

        f_list = []
        # f_list.append(load_obj('best_surro'))  # optimize using surrogate model
        f_list.append(ackley(dimension))  # optimize using real function
        opt = optimization(lower_bound, upper_bound, f_list, max_gen=200, pop_size=10)
        opt.optimize()
        opt.plot()

    else: # multi object demo
        dimension = 30

        lower_bound = [0]*dimension
        upper_bound = [1]*dimension

        f_list = []
        # f_list.append(load_obj('best_surro'))  # optimize using surrogate model
        f_list.append(ZDT1_obj0())  # optimize using real function
        f_list.append(ZDT1_obj1())  # optimize using real function
        # f_list.append(DTLZ1_obj2())  # optimize using real function
        opt = optimization(lower_bound, upper_bound, f_list, max_gen=500, pop_size=100)
        opt.optimize()