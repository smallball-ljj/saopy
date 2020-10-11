# ==================================================
# author:luojiajie
# ==================================================

import os
import csv
import numpy as np
import matplotlib.pyplot as plt

from saopy.optimization import *
from saopy.exploration import *
from saopy.exploitation import *


class ass():
    def __init__(self, surro_list, outer_iter, plot_flag=0, plot_y_range=[]):
        self.surro_list=surro_list # a list of surrogate model
        self.outer_iter=outer_iter # outer iteration index
        self.plot_flag=plot_flag # plot or not for exploration and exploitation
        self.plot_y_range=plot_y_range # plot y range for exploitation

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


    def generate_new_X(self,number_optimized_X,number_exploitation_X,number_exploration_X):
        # optimization
        opt = optimization(self.surro_list[0].lower_bound, self.surro_list[0].upper_bound, self.surro_list, max_gen=2000, pop_size=number_optimized_X)
        last_population = opt.optimize()
        if self.plot_flag==0:
            opt.plot(self.outer_iter)
        if os.path.exists('Result_'+str(self.outer_iter)) == False:
            os.makedirs('Result_'+str(self.outer_iter))
        np.savetxt('Result_'+str(self.outer_iter)+'/ObjV.csv', last_population.ObjV, delimiter=',')
        np.savetxt('Result_'+str(self.outer_iter)+'/Phen.csv', last_population.Phen, delimiter=',')

        # exploitation
        exploit = exploitation(self.surro_list)
        exploit.generate_exploitation_X(number_exploitation_X)
        if self.plot_flag==1:
            exploit.plot(self.plot_y_range,self.outer_iter)

        # exploration
        # explorat = exp_fullfactorial()
        explorat = exp_random_lhs()
        explorat.load_data(self.surro_list[0].lower_bound, self.surro_list[0].upper_bound, file_X='X.csv')
        explorat.normalize_X()
        explorat.generate_candidate_points(1000000)
        explorat.generate_exploration_X(number_exploration_X, self.outer_iter, self.plot_flag)

        # stack three categories and remove duplicate points
        self.load_data()
        self.new_X = np.vstack((self.optimized_X, self.exploitation_X))
        self.new_X = np.vstack((self.new_X, self.exploration_X))
        if self.new_X.shape[0] != np.unique(self.new_X, axis=0).shape[0]:
            print('there exist duplicate samples between different categories')
            self.new_X=np.unique(self.new_X, axis=0)

        self.X = self.surro_list[0].X
        X_tmp = np.vstack((self.X, self.new_X))
        if X_tmp.shape[0] != np.unique(X_tmp, axis=0).shape[0]:
            print('there exist duplicate samples between new X and X')
            self.remove_duplicate_new_X()

        np.savetxt('X_new.csv', self.new_X, delimiter=',')


    def load_data(self):
        if len(self.surro_list)==1: # single objective, stack the best variable, for validation and comparison with best_ObjV.csv
            best_var = self.read_csv_to_np('best_var.csv')
            self.optimized_X = np.array([best_var[self.outer_iter, :]])
        else:  # multi objective, stack all the samples on the pareto front
            self.optimized_X = self.read_csv_to_np('Result/Phen.csv')
            self.optimized_X = np.unique(self.optimized_X,axis=0)  # note: the samples in 'Result/Phen.csv' may be duplicate, so we remove it, otherwise calculate_distance_phi() will get infinite value

        self.exploitation_X = self.read_csv_to_np('X_exploitation.csv')
        self.exploration_X = self.read_csv_to_np('X_exploration.csv')

        # note: the algorithm for exploitation and exploration already ensure non-duplicate samples, we double check it if there still exist other bugs
        if self.exploitation_X.shape[0] != np.unique(self.exploitation_X, axis=0).shape[0]:
            raise RuntimeError('there exist duplicate samples in exploitation_X')
        if self.exploration_X.shape[0] != np.unique(self.exploration_X, axis=0).shape[0]:
            raise RuntimeError('there exist duplicate samples in exploration_X')


    def remove_duplicate_new_X(self):
        i = 0
        while True:
            tmp_X = np.vstack((self.X, self.new_X[i, :]))
            if tmp_X.shape[0] != np.unique(tmp_X, axis=0).shape[0]:
                self.new_X = np.delete(self.new_X, i, axis=0)
                i -= 1
            i += 1
            if i == self.new_X.shape[0]: break



    def plot(self, surro, y_range=[], outer_iter=0):
        """
        plot scatter of self.exploration_X, self.exploitation_X, self.optimized_X, self.new_X

        :param surro: surrogate model, used for plotting response surface
        :param y_range: 1. if y_range==[], plot y according to calculated max range of y
                        2. plot y acoording to given range: y_range=[lower boundary, upper boundary, step]
                        e.g. y_range=[0,10,1]
        :param outer_iter: outer iteration index (used for saving the plot of new updated surrogate model)
        """
        surro.point_max_error = surro.inverse_normalize_X(np.array([surro.point_max_error])) # inverse normalize point of max error
        surro.point_max_error.resize(surro.point_max_error.shape[1])

        if os.path.exists('plot') == False:
            os.makedirs('plot')
        dimension=self.optimized_X.shape[1]
        # ==================================================
        # 1D
        # ==================================================
        if dimension == 1:
            plt,p0 = surro.plot(surro.lower_bound, surro.upper_bound, y_range, show_flag=0)  # get plt from surrogate model
            p1 = plt.scatter(self.optimized_X[:, 0], np.zeros((self.optimized_X.shape[0], 1)), s=50, marker='o',c='none',edgecolors='black')
            p2 = plt.scatter(self.exploitation_X[:, 0], np.zeros((self.exploitation_X.shape[0], 1)), s=50, marker='^', c='none',edgecolors='black')
            p3 = plt.scatter(self.exploration_X[:, 0], np.zeros((self.exploration_X.shape[0], 1)), s=50, marker='s', c='none',edgecolors='black')
            # p4 = plt.scatter(self.new_X[:, 0], np.zeros((self.new_X.shape[0], 1)), 25, marker='o', c='black')

            # legend
            font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15, }
            plt.legend(handles=[p0, p1, p2, p3], labels=['original X', 'optimized X', 'exploitation X', 'exploration X'], loc='best', edgecolor='black', prop=font)

        # ==================================================
        # multi dimension
        # ==================================================
        elif dimension >= 2:
            plt, p0 = surro.plot(surro.lower_bound, surro.upper_bound, y_range,
                                      rest_var=surro.point_max_error,
                                      show_flag=0)  # get plt and scatter of all samples from surrogate model

            for i in range(dimension):
                for j in range(dimension - 1, i, -1):
                    col_ind = i  # column index
                    row_ind = -j + dimension - 1  # row index
                    ind = row_ind * (dimension - 1) + col_ind + 1  # subplot index
                    plt.subplot(dimension - 1, dimension - 1, ind)  # locate subplot
                    p1, p2, p3 = self.plot_md(i, j)  # plot response surface in subplot
                    # plot axis label
                    fontXY = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }  # xy label size
                    axisfont = 20  # axis number size
                    if row_ind == 0 and col_ind == 0:  # first subplot, plot both axis
                        ax = plt.gca()
                        ax.xaxis.set_ticks_position('top')
                        ax.xaxis.set_label_position('top')
                        plt.xlabel('x' + str(i), fontXY)
                        plt.ylabel('x' + str(j), fontXY)
                    elif row_ind == 0 and col_ind != 0:  # first row, plot x axis label
                        plt.yticks([])
                        ax = plt.gca()
                        ax.xaxis.set_ticks_position('top')
                        ax.xaxis.set_label_position('top')
                        plt.xlabel('x' + str(i), fontXY)
                    elif col_ind == 0 and row_ind != 0:  # first column, plot y axis label
                        plt.xticks([])
                        plt.ylabel('x' + str(j), fontXY)
                    else:  # other subplot, do not show x,y axis label
                        plt.xticks([])
                        plt.yticks([])
                    plt.xlim(surro.lower_bound[i], surro.upper_bound[i])  # x range
                    plt.ylim(surro.lower_bound[j], surro.upper_bound[j])  # x range
                    plt.tick_params(labelsize=axisfont)  # axis number size

            if dimension > 2:  # dimension>2 case, plot legend in subplot
                plt.subplot(2, dimension - 1, 2 * (dimension - 1))  # locate color bar subplot
            # legend
            font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15, }
            plt.legend(handles=[p0, p1, p2, p3],labels=['original X', 'optimized X', 'exploitation X', 'exploration X'], loc='best',edgecolor='black', prop=font)
        # plt.show()
        plt.savefig('plot/adaptive_sequential_sampling_'+str(outer_iter)+'.eps')
        plt.close()


    def plot_md(self, x1_arg, x2_arg):
        p1 = plt.scatter(self.optimized_X[:, x1_arg], self.optimized_X[:, x2_arg], s=50, marker='o',c='none',edgecolors='black')
        p2 = plt.scatter(self.exploitation_X[:, x1_arg], self.exploitation_X[:, x2_arg], s=50, marker='^', c='none',edgecolors='black')
        p3 = plt.scatter(self.exploration_X[:,x1_arg], self.exploration_X[:,x2_arg], s=50, marker='s',c='none',edgecolors='black')
        # p4 = plt.scatter(self.new_X[:, x1_arg], self.new_X[:, x2_arg], 25, marker='o', c='black')
        return p1, p2, p3