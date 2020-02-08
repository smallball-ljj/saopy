# ==================================================
# author:luojiajie
# ==================================================

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


class exploration():
    """
    base class of exploration
    """
    def __init__(self):
        self.X = None # got from load_data(). all samples X
        self.normalized_X_original = None  # got from normalize_X(). all original normalized samples X
        self.normalized_X = None # got from normalize_X(). all normalized samples X and will be updated by update_X() once new point is generated
        self.points = None # candidate_points. got from generate_candidate_points()
        self.exploration_X = [] # additional samples generated by exploration


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


    def load_data(self,lower_bound,upper_bound,file_X='X.csv'):
        """
        lower_bound,upper_bound: range of X, used for normalizaiton
        e.g. for 2 dimension, lower_bound=[5,10] upper_bound=[10,100]
        the bound should be consistent with the one generated using sampling_plan.py
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.X = self.read_csv_to_np(file_X)


    def normalize_X(self):
        """
        normalize X to [0,1] according to lower_bound and upper_bound (e.g. for 2 dimension, lower_bound=[5,10] upper_bound=[10,100])
        the bound should be consistent with the one generated using sampling_plan.py
        """
        X_n = self.X.copy()
        for i in range(X_n.shape[1]):
            X_n[:, i] = (X_n[:, i] - self.lower_bound[i]) / (self.upper_bound[i] - self.lower_bound[i])
        self.normalized_X = X_n
        self.normalized_X_original = self.normalized_X.copy()


    def inverse_normalize_X(self,X):
        X_n = X.copy()
        for i in range(X_n.shape[1]):
            X_n[:,i] = X_n[:,i]*(self.upper_bound[i] - self.lower_bound[i])+self.lower_bound[i]
        return X_n


    def generate_candidate_points(self, number):
        """
        detailed method defined in subclass, which will call methods in sampling_plan.py
        :param number: number of samples
        (for Latin hypercube sampling, this means total number of samples)
        (for full factorial sampling, this means number of points for each dimension)
        """
        pass


    def cal_min_distance(self):
        """
        calculate the minimum distance between all candidate points and X
        """
        self.min_dis = np.zeros((self.points.shape[0],1)) # initialize minimum distance
        for i in range(self.points.shape[0]):
            dis=np.sum((self.normalized_X-self.points[i])**2,axis=1)**0.5 # calculate the distance between the i th points and all the X. note: matrix calculation is much faster than using 'for'
            self.min_dis[i] = dis.min() # select the minimum distance


    def get_new_point(self):
        """
        choose the point in all candidate points, which has the largest min_dis
        """
        self.new_point = self.points[self.min_dis.argmax()]


    def update_X(self):
        """
        stack new point to the end of all original samples: self.normalized_X
        """
        self.normalized_X=np.vstack((self.normalized_X,self.new_point))


    def update_min_distance(self):
        """
        update minimum distance: if the distance between new point and candidate point is smaller than the original minimum distance, update it
        :return:
        """
        new_dis=np.sum((self.new_point - self.points) ** 2, axis=1)**0.5 # calculate the distance between the new point and all the candidate points. note: matrix calculation is much faster than using 'for'
        new_dis=new_dis.reshape((new_dis.shape[0],1))
        diff=new_dis-self.min_dis # calculate the difference between original min_dis and new_dis
        diff[diff>0]=0 # assgin the value in diff to 0 if new_dis>self.min_dis, so that this row of self.min_dis won't update
        self.min_dis=diff+self.min_dis # update minimum distance


    def generate_exploration_X(self,number,outer_iter=0,plot_flag=0):
        """
        :param number: total number of additional samples to generate by exploration
        :param outer_iter: outer iteration index (used for saving the plot of new updated surrogate model)
        :param plot_flag: plot or not
        """
        for i in range(number):
            if i == 0:
                self.cal_min_distance()
            else:
                self.update_min_distance()
            self.get_new_point()
            self.update_X()
            if i == 0:
                self.exploration_X = np.array([self.new_point])
            else:
                self.exploration_X = np.vstack((self.exploration_X,np.array([self.new_point])))

            if plot_flag==1:
                try: # in case you use exp_random_lhs and 2D, and the total number of candidate points is not square number
                    # self.plot(i, [0, 0.03, 0.003],outer_iter)
                    self.plot(i,[],outer_iter)
                except:
                    print('exploration plot error')

        # output
        self.exploration_X=self.inverse_normalize_X(self.exploration_X)
        np.savetxt('X_exploration.csv', self.exploration_X, delimiter=',')


    def plot(self, i, y_range=[], outer_iter=0):
        """
        plot self.min_dis
        and scatter of all samples X and new point
        only available for 1D or 2D now

        :param i: the i th plot of new generated point
        :param y_range: 1. if y_range==[], plot y according to calculated max range of y
                        2. plot y acoording to given range: y_range=[lower boundary, upper boundary, step]
                        e.g. y_range=[0,10,1]
        :param outer_iter: outer iteration index (used for saving the plot of new updated surrogate model)
        """
        if os.path.exists('plot') == False:
            os.makedirs('plot')
        # ==================================================
        # 1D
        # ==================================================
        if self.points.shape[1] == 1:
            # plot
            fig = plt.figure(figsize=(15, 7.5))  # figure size
            p = plt.plot(self.points, self.min_dis, linestyle='-', linewidth=3, c='black', label='intersite distance')
            plt.xlim(0, 1)  # x range
            if y_range==[]:
                pass
            else:
                plt.ylim(y_range[0], y_range[1]) # y range
            # legend
            # font_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
            # plt.legend(loc='best', edgecolor='black', prop=font_legend)
            # xy label size
            font_xy = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
            plt.xlabel('x', font_xy)
            plt.ylabel('intersite distance', font_xy)
            # coordinate font
            plt.tick_params(labelsize=20)

            plt.scatter(self.normalized_X_original[:, 0], np.zeros((self.normalized_X_original.shape[0], 1)), s=5, marker='o', c='black') # plot scatter of all original samples X
            plt.scatter(self.exploration_X[:, 0], np.zeros((self.exploration_X.shape[0],1)), s=50, marker='s', c='none', edgecolors='black') # plot scatter of all exploration X
            plt.scatter(self.new_point[0], np.zeros((1,1)), s=50, marker='s', c='black') # plot new point generated this time

            # plt.show()
            plt.savefig('plot/exploration_' + str(outer_iter) + '_' + str(i) + '.eps')
            plt.close()

        # ==================================================
        # 2D
        # ==================================================
        elif self.points.shape[1] == 2:
            X1 = self.points[:, 0]
            X2 = self.points[:, 1]

            plotgrid=int(self.points.shape[0] ** 0.5)
            X1 = X1.reshape((plotgrid, plotgrid))
            X2 = X2.reshape((plotgrid, plotgrid))

            y = self.min_dis.reshape((plotgrid, plotgrid)).copy()

            colorslist = ['blue', 'aqua', 'yellow', 'red']  # color of color bar
            cmaps = colors.LinearSegmentedColormap.from_list('mylist', colorslist, N=3000)

            fig = plt.figure(figsize=(10, 7.5))  # figure size

            if y_range == []:  # plot contour according to y range
                cont = plt.contourf(X1, X2, y, 50, cmap=cmaps)
                cbar = plt.colorbar(cont)  # plot color bar

            else:  # plot contour according to given range
                y[y < y_range[0]] = y_range[0]  # assgin any y, which is smaller than the given lower boundary, to lower boundary
                y[y > y_range[1]] = y_range[1]  # assgin any y, which is larger than the given upper boundary, to upper boundary

                lim = np.arange(y_range[0], y_range[1] + 0.0000001, (y_range[1] - y_range[0]) / 50)
                cont = plt.contourf(X1, X2, y, lim, cmap=cmaps)  # plot contour
                cbar = plt.colorbar(cont,ticks=np.arange(y_range[0], y_range[1] + 1, y_range[2]).tolist())  # plot color bar

            # ==================================================
            # xy label size
            fontXY = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
            # colorbar label size
            fontC = {'family': 'Times New Roman', 'weight': 'normal', 'size': 25, }
            axisfont = 20  # axis number size
            plt.xlabel('x1', fontXY)
            plt.ylabel('x2', fontXY)
            plt.tick_params(labelsize=axisfont)  # axis number size
            cbar.ax.set_ylabel('intersite distance', fontC)  # color bar label
            cbar.ax.tick_params(labelsize=axisfont)  # color bar axis number size

            plt.scatter(self.normalized_X_original[:, 0], self.normalized_X_original[:, 1], s=5, marker='o', c='black') # plot scatter of all original samples X
            plt.scatter(self.exploration_X[:, 0], self.exploration_X[:, 1], s=50, marker='s', c='none', edgecolors='black') # plot scatter of all exploration X
            plt.scatter(self.new_point[0], self.new_point[1], s=50, marker='s', c='black') # plot new point generated this time
            plt.xlim(0,1)
            plt.ylim(0,1)

            # plt.show()
            plt.savefig('plot/exploration_' + str(outer_iter) + '_' + str(i) + '.eps')
            plt.close()



class exp_fullfactorial(exploration):
    def generate_candidate_points(self,number):
        n=number
        d=self.X.shape[1]
        ix = (slice(0, 1, n*1j),) * d
        self.points = np.mgrid[ix].reshape(d, n**d).T

class exp_random_lhs(exploration):
    def generate_candidate_points(self,number):
        n=number
        d=self.X.shape[1]
        Edges = 1
        X = np.zeros((n,d))
        for i in range(0,d):
            X[:,i] = np.transpose(np.random.permutation(np.arange(1,n+1,1)))   # e.g. if total number of samples: n=5, then  X[:,i]=[1,3,2,5,4]
        if Edges == 1:
            X = (X-1)/(n-1)
        else:
            X = (X-0.5)/n
        self.points=X



# e.g.
if __name__ == '__main__':
    lower_bound = [-32.768, -32.768]
    upper_bound = [32.768, 32.768]

    explorat = exp_fullfactorial()
    # explorat=exp_random_lhs()
    explorat.load_data(lower_bound, upper_bound, file_X=r'demo\single_obj\ackley_2D\X.csv')
    explorat.normalize_X()
    explorat.generate_candidate_points(100)
    explorat.generate_exploration_X(10, outer_iter=0,plot_flag=1)