# ==================================================
# author:luojiajie
# ==================================================

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Note: the rows in 'Result/Phen.csv' 'X_exploitation.csv' 'X_exploration.csv' must larger than 1, or the data loaded by np.loadtxt will have bug, which is not consistent with the original data structure,
this bug can be fixed later
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from itertools import combinations
from multiprocessing import Pool


class ass():
    def read_csv_to_np(self,file_name):
        """
        read data from .csv and convert to numpy.array

        Examples:
            a=read_csv_to_np('test.csv'):
        """
        data = np.loadtxt(file_name, delimiter=',')
        try:  # note: when the dimension of samples is 1, the loaded data is in shape(n,) so we resize it to shape(n,1)
            data.shape[1]
        except:
            data.resize((data.shape[0], 1))
        return data


    def load_data(self,lower_bound,upper_bound):
        """
        lower_bound,upper_bound: range of X, used for normalizaiton
        e.g. for 2 dimension, lower_bound=[5,10] upper_bound=[10,100]
        the bound should be consistent with the one generated using sampling_plan.py
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.optimized_X = self.read_csv_to_np('Result/Phen.csv')
        self.optimized_X = np.unique(self.optimized_X, axis=0) # note: the samples in 'Result/Phen.csv' may be duplicate, so we remove it, otherwise calculate_distance_phi() will get infinite value
        self.exploitation_X = self.read_csv_to_np('X_exploitation.csv')
        self.exploration_X = self.read_csv_to_np('X_exploration.csv')

        # note: the algorithm for exploitation and exploration already ensure non-duplicate samples, we double check it if there still exist other bugs
        if self.exploitation_X.shape[0] != np.unique(self.exploitation_X, axis=0).shape[0]:
            raise RuntimeError('there exist duplicate samples in exploitation_X')
        if self.exploration_X.shape[0] != np.unique(self.exploration_X, axis=0).shape[0]:
            raise RuntimeError('there exist duplicate samples in exploration_X')


    def normalize_X(self,X):
        """
        normalize X to [0,1] according to lower_bound and upper_bound
        """
        X_n = X.copy()
        for i in range(X_n.shape[1]):
            X_n[:, i] = (X_n[:, i] - self.lower_bound[i]) / (self.upper_bound[i] - self.lower_bound[i])
        return X_n

    def inverse_normalize_X(self,X):
        X_n = X.copy()
        for i in range(X_n.shape[1]):
            X_n[:,i] = X_n[:,i]*(self.upper_bound[i] - self.lower_bound[i])+self.lower_bound[i]
        return X_n


    def normalize_all(self):
        self.normalized_optimized_X = self.normalize_X(self.optimized_X)
        self.normalized_exploitation_X = self.normalize_X(self.exploitation_X)
        self.normalized_exploration_X = self.normalize_X(self.exploration_X)


    def generate_new_X(self,number_optimized_X,number_exploitation_X,number_exploration_X,parallel_num):
        """
        first get all the combinations of each category, then combine each combination together, then calculate distance phi, finally select the combination, which has the min distance phi
        e.g. suppose the total number of samples in each category is 10, and the number of samples to select in each category is 5,
        then the number of all the combinations of each category is 10C5, the number of all the combinations for the 3 categories is (10C5)^3

        :param number_optimized_X,number_exploitation_X,number_exploration_X: the number of samples of new_X to be selected in each category
        :param parallel_num: total parallel process number (must >=1 )
        """
        if number_optimized_X > self.optimized_X.shape[0]:
            print('the number for optimized X is larger than non-duplicate samples:', self.optimized_X.shape[0], ',use', self.optimized_X.shape[0])
            number_optimized_X=self.optimized_X.shape[0]

        pool = Pool(processes=parallel_num) # create processes

        comb_list_optimized_X = list(combinations(range(self.normalized_optimized_X.shape[0]), number_optimized_X)) # all combinations of optimized_X
        comb_list_exploitation_X  = list(combinations(range(self.normalized_exploitation_X.shape[0]), number_exploitation_X)) # all combinations of exploitation_X
        comb_list_exploration_X = list(combinations(range(self.normalized_exploration_X.shape[0]), number_exploration_X)) # all combinations of exploration_X

        total_combinations_to_calculate=len(comb_list_optimized_X)*len(comb_list_exploitation_X)*len(comb_list_exploration_X)
        combinations_to_calculate_per_process=total_combinations_to_calculate//parallel_num
        print('calculation time estimation:', 0.0022*combinations_to_calculate_per_process, 's')

        parallel_process = [] # parallel process list
        for i in range(parallel_num): # append parallel process
            star_ind=i*combinations_to_calculate_per_process
            if i==parallel_num-1:
                end_ind=total_combinations_to_calculate-1
            else:
                end_ind=(i+1)*combinations_to_calculate_per_process-1
            parallel_process.append(pool.apply_async(self.get_dis_and_new_X,(number_optimized_X,number_exploitation_X,number_exploration_X,comb_list_optimized_X,comb_list_exploitation_X,comb_list_exploration_X,star_ind,end_ind,)))  # append parallel process
        # pool.close() #
        # pool.join() # wait all the process to finish. this has bug when running on HPC, maybe due to the version, has not fixed yet

        dis = 0
        for p in parallel_process:
            dis_temp, temp_X=p.get()
            if dis_temp > dis:
                dis = dis_temp
                self.new_X = temp_X.copy()

        pool.close()
        # all calculation complete, save new_X
        self.new_X=self.inverse_normalize_X(self.new_X)
        np.savetxt('X_new.csv', self.new_X, delimiter=',')


    def get_dis_and_new_X(self,number_optimized_X,number_exploitation_X,number_exploration_X,comb_list_optimized_X,comb_list_exploitation_X,comb_list_exploration_X,star_ind,end_ind):
        number_total = number_optimized_X + number_exploitation_X + number_exploration_X  # total number of samples for new_X
        new_X = np.zeros((number_total, self.normalized_optimized_X.shape[1]))  # initialize new_X
        temp_X = np.zeros((number_total, self.normalized_optimized_X.shape[1]))  # initialize temp_X, which is one combination of samples of all the combinations, so that it can be pass to calculate_distance_phi() to calculate distance
        dis = 0  # initialize 1/distance_phi. note: the space has better fillness with larger dis
        break_flag=0 # break flag for all loop

        count_optimized_X = 0 # count for loop comb_list_optimized_X
        for i in comb_list_optimized_X:
            if break_flag: break
            count_optimized_X += 1
            optimized_X_loop_end_ind = count_optimized_X * len(comb_list_exploitation_X) * len(comb_list_exploration_X) # calculate the index at the end of the total loop, if this index is larger than star_ind, execute the following loop
            if optimized_X_loop_end_ind > star_ind:
                ind_optimized = 0
                for ii in i:
                    temp_X[ind_optimized] = self.normalized_optimized_X[ii]
                    ind_optimized += 1

                count_exploitation_X = 0  # count for loop comb_list_exploitation_X
                for j in comb_list_exploitation_X:
                    if break_flag: break
                    count_exploitation_X += 1
                    exploitation_X_loop_end_ind = (count_optimized_X-1) * len(comb_list_exploitation_X) * len(comb_list_exploration_X) + count_exploitation_X * len(comb_list_exploration_X)
                    if exploitation_X_loop_end_ind > star_ind:
                        ind_exploitation = number_optimized_X
                        for jj in j:
                            temp_X[ind_exploitation] = self.normalized_exploitation_X[jj]
                            ind_exploitation += 1

                        count_exploration_X = 0 # count for loop comb_list_exploration_X
                        for k in comb_list_exploration_X:
                            count_exploration_X += 1
                            exploration_X_loop_end_ind = (count_optimized_X-1) * len(comb_list_exploitation_X) * len(comb_list_exploration_X) + (count_exploitation_X-1) * len(comb_list_exploration_X) + count_exploration_X
                            if exploration_X_loop_end_ind > star_ind:
                                ind_exploration = number_optimized_X + number_exploitation_X
                                for kk in k:
                                    temp_X[ind_exploration]=self.normalized_exploration_X[kk]
                                    ind_exploration += 1

                                if temp_X.shape[0] != np.unique(temp_X, axis=0).shape[0]:
                                    print('there exist duplicate samples between different categories')
                                    dis_temp=0
                                else:
                                    dis_temp=1/self.calculate_distance_phi(temp_X)

                                if dis_temp>dis:
                                    dis = dis_temp
                                    new_X = temp_X.copy() # note: must use copy(), or new_X will change with temp_X, which means they share the same memory if don't use copy()

                            if exploration_X_loop_end_ind > end_ind:
                                break_flag=1
                                break

        return dis, new_X


    # the following distance calculation code are from sampling_plan.py
    def calculate_distance_phi(self,X,q=2,p=2):
        """
        Calculates the sampling plan quality criterion of Morris and Mitchell

        Inputs:
            X - Sampling plan
            q - exponent used in the calculation of the metric (default = 2)
            p - the distance metric to be used (p=1 rectangular - default , p=2 Euclidean)

        Output:
            Phiq - sampling plan 'space-fillingness' metric. the smaller Phiq, the space has better fillness
        """
        #calculate the distances between all pairs of points (using the p-norm) and build multiplicity array J
        J,d = self.jd(X,p)
        if d.min()==0:
            print(d)
        #the sampling plan quality criterion
        Phiq = (np.sum(J*(d**(-q))))**(1.0/q)
        return Phiq

    def jd(self, X,p=2):
        """
        Computes the distances between all pairs of points in a sampling plan X using the p-norm, sorts them in ascending order and removes multiple occurences.

        Inputs:
            X-sampling plan being evaluated
            p-distance norm (p=1 rectangular-default, p=2 Euclidean)
        Output:
            J-multiplicity array (that is, the number of pairs with the same distance value)
            distinct_d-list of distinct distance values
        """
        #number of points in the sampling plan
        n = np.size(X[:,0])

        #computes the distances between all pairs of points
        d = np.zeros((n*(n-1)//2))

        # ind=0
        # for i in range(n-1):
        #     for j in range(i+1,n):
        #         d[ind] = np.linalg.norm((X[i,:] - X[j,:]),p)
        # ind+=1

        #an alternative way of the above loop
        list = [(i,j) for i in range(n-1) for j in range(i+1,n)]
        for k,l in enumerate(list):
            d[k] = np.linalg.norm((X[l[0],:]-X[l[1],:]),p)

        #remove multiple occurences and sort in ascending order
        distinct_d, J = np.unique(d, return_counts=True)

        return J, distinct_d



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
            p4 = plt.scatter(self.new_X[:, 0], np.zeros((self.new_X.shape[0], 1)), 25, marker='o', c='black')

            # legend
            font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15, }
            plt.legend(handles=[p0, p1, p2, p3, p4], labels=['original X', 'optimized X', 'exploitation X', 'exploration X', 'new X'], loc='best', edgecolor='black', prop=font)

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
                    p1, p2, p3, p4 = self.plot_md(i, j)  # plot response surface in subplot
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
            plt.legend(handles=[p0, p1, p2, p3, p4],labels=['original X', 'optimized X', 'exploitation X', 'exploration X', 'new X'], loc='best',edgecolor='black', prop=font)
        # plt.show()
        plt.savefig('plot/adaptive_sequential_sampling_'+str(outer_iter)+'.eps')
        plt.close()


    def plot_md(self, x1_arg, x2_arg):
        p1 = plt.scatter(self.optimized_X[:, x1_arg], self.optimized_X[:, x2_arg], s=50, marker='o',c='none',edgecolors='black')
        p2 = plt.scatter(self.exploitation_X[:, x1_arg], self.exploitation_X[:, x2_arg], s=50, marker='^', c='none',edgecolors='black')
        p3 = plt.scatter(self.exploration_X[:,x1_arg], self.exploration_X[:,x2_arg], s=50, marker='s',c='none',edgecolors='black')
        p4 = plt.scatter(self.new_X[:, x1_arg], self.new_X[:, x2_arg], 25, marker='o', c='black')
        return p1, p2, p3, p4