# ==================================================
# author:luojiajie
# ==================================================

import os
import csv
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


def save_obj(obj, save_dir):
    """
    save the entire object to <save_dir>
    :param obj      : any object
    :param save_dir : save directory
    Example:
        s=ANN()
        save_obj(s,'model_ann')
    """
    pickle.dump(obj, open(save_dir, 'wb'))


def load_obj(load_dir):
    """
    load entire object from <load_dir?
    :param load_dir : load directory
    :return         : any object
    Example:
        s=load_model('model_ann')
    """
    return pickle.load(open(load_dir, 'rb'))


class surrogate_model():
    """
    base class for all surrogate models
    """
    def __init__(self):
        self.X = None # array, got from load_data()
        self.y = None # array, got from load_data()

        self.normalized_X = None # array, got from normalize_all()
        self.normalized_y = None # array, got from normalize_all()

        self.normalized_fold_X = None # list[array,array...], got from calling <divide_method.random(surro,num_fold=3)> or other divide methods
        self.normalized_fold_y = None # and it will seperate self.normalized_X and self.normalized_y into n fold. then, call <cross_valid()> and it will use n-1 fold as training data and 1 fold as testing data.

        self.y_min = None # max of y, got from load_data()
        self.y_max = None # min of y, got from load_data()

        self.point_max_error = None # got from cross_validation()
        self.max_test_error = None # got from cross_validation(). as the level of exploitation

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


    def load_data(self,lower_bound,upper_bound,file_X='X.csv',file_y='y.csv'):
        """
        lower_bound,upper_bound: range of X, used for normalizaiton
        e.g. for 2 dimension, lower_bound=[5,10] upper_bound=[10,100]
        the bound should be consistent with the one generated using sampling_plan.py
        """
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound

        self.X = self.read_csv_to_np(file_X) # load X
        self.y = self.read_csv_to_np(file_y) # load y

        # get ymin and ymax for normailzation
        self.y_min=self.y.min()
        self.y_max=self.y.max()


    def normalize_X(self,X):
        """
        normalize X to [0,1] according to lower_bound and upper_bound (e.g. for 2 dimension, lower_bound=[5,10] upper_bound=[10,100])
        the bound should be consistent with the one generated using sampling_plan.py
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

    def normalize_y(self,y):
        """
        normalize y to [0,1] according to maximum and minimum of self.y
        """
        return (y-self.y_min)/(self.y_max-self.y_min)

    def inverse_normalize_y(self,y):
        return y*(self.y_max-self.y_min)+self.y_min


    def normalize_all(self):
        self.normalized_X=self.normalize_X(self.X)
        self.normalized_y=self.normalize_y(self.y)


    def train(self):
        """
        detailed train process defined in specific models
        """
        pass


    def calculate(self,X):
        """
        detailed calculate process defined in specific models

        :param  X: numpy array, with shape(number,dimension)
        :return y: numpy array, with shape(number,1)
        """
        pass


    def plot(self, lower_bound, upper_bound, y_range=[], x_label=None, y_label='y', rest_var=None, show_flag=1, sample_flag=1, outer_iter=0):
        """
        plot response surface

        :param lower_bound: lower boundary of X   e.g. for 2D: lower_bound=[-1,-2], for 1D: lower_bound=[-1]
        :param upper_bound: upper boundary of X   e.g. for 2D: upper_bound=[3,4], for 1D: upper_bound=[3]
        :param y_range: 1. if y_range==[], plot y according to calculated max range of y (default)
                        2. plot y acoording to given range: y_range=[lower boundary, upper boundary, step]
                        e.g. y_range=[0,10,1]
        :param x_label: 1. if x_label==None, plot x label like 'x0, x1, x2 ...' (default)
                        2. plot x label acoording to given label
                        e.g. x_label=['length', 'height']
        :param rest_var: available for dimension>=3. note: the response surface is plotted over the two selected variables, so the rest varible will keep the same value
                          if rest_var==None, the rest variables are set as the mid point of lower_bound and upper_bound by default
        :param show_flag: if show_flag==1, show the plot (default)
                          if show_flag==0, do not show the plot (which will return plt, for other class to call it (e.g. see exploitation.py)
        :param sample_flag: if sample_flag==1, show all the samples (default)
                            if sample_flag==0, do not show all the samples
        :param outer_iter: outer iteration index (used for saving the plot of new updated surrogate model)
        """
        if os.path.exists('plot') == False:
            os.makedirs('plot')
        dimension=len(lower_bound)
        # ==================================================
        # 1D
        # ==================================================
        if dimension==1:
            plotgrid = 1000  # the number of grid points
            X = np.linspace(lower_bound[0], upper_bound[0], plotgrid)
            X.resize((X.shape[0], 1))
            y = self.calculate(X)

            # plot
            fig = plt.figure(figsize=(15, 7.5)) # figure size
            p=plt.plot(X, y, linestyle='--', linewidth=3, c='black',label='surrogate model')
            plt.xlim(lower_bound[0], upper_bound[0]) # x range
            if y_range==[]:
                pass
            else:
                plt.ylim(y_range[0], y_range[1]) # y range
            # legend
            # font_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
            # plt.legend(loc='best',edgecolor='black', prop=font_legend)
            # xy label size
            font_xy = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
            plt.xlabel('x', font_xy)
            plt.ylabel('y', font_xy)
            # coordinate font
            plt.tick_params(labelsize=20)

            # plot scatter of all samples X
            if sample_flag==1:
                p1 = plt.scatter(self.X, np.zeros((self.X.shape[0],1)), s=5, marker='o', c='black')
                plt.xlim(lower_bound[0], upper_bound[0])  # x range
                font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
                plt.legend(handles=[p1], labels=['training samples'],loc='best', edgecolor='black', prop=font)


        # ==================================================
        # multi dimension
        # ==================================================
        elif dimension >= 2:
            # assgin x label if not given
            if x_label == None:
                x_label=[]
                for i in range(dimension):
                    x_label.append('x'+str(i))

            fig = plt.figure(figsize=(10, 7.5))  # figure size
            for i in range(dimension):
                for j in range(dimension-1, i, -1):
                    col_ind = i # column index
                    row_ind = -j + dimension - 1 # row index
                    ind = row_ind * (dimension - 1) + col_ind + 1 # subplot index
                    plt.subplot(dimension - 1, dimension - 1, ind) # locate subplot
                    cont, p1=self.plot_md(i, j, lower_bound, upper_bound, y_range, rest_var, show_flag_md=0, sample_flag=sample_flag) # plot response surface in subplot
                    # plot axis label
                    fontXY = {'family': 'Times New Roman', 'weight': 'normal', 'size': 35, }  # xy label size
                    axisfont = 25  # axis number size
                    if row_ind == 0 and col_ind == 0:  # first subplot, plot both axis
                        ax = plt.gca()
                        ax.xaxis.set_ticks_position('top')
                        ax.xaxis.set_label_position('top')
                        plt.xlabel(x_label[i], fontXY)
                        plt.ylabel(x_label[j], fontXY)
                    elif row_ind == 0 and col_ind != 0:  # first row, plot x axis label
                        plt.yticks([])
                        ax = plt.gca()
                        ax.xaxis.set_ticks_position('top')
                        ax.xaxis.set_label_position('top')
                        plt.xlabel(x_label[i], fontXY)
                    elif col_ind == 0 and row_ind != 0:  # first column, plot y axis label
                        plt.xticks([])
                        plt.ylabel(x_label[j], fontXY)
                    else:  # other subplot, do not show x,y axis label
                        plt.xticks([])
                        plt.yticks([])
                    plt.xlim(lower_bound[i], upper_bound[i])  # x range
                    plt.ylim(lower_bound[j], upper_bound[j])  # x range
                    plt.tick_params(labelsize=axisfont)  # axis number size

            if dimension > 2: # dimension>2 case, plot color bar in subplot
                plt.subplot(2, dimension - 1, 2 * (dimension - 1)) # locate color bar subplot
                plt.axis('off') # do not plot axis

            if y_range == []:  # plot according to y range
                cbar = plt.colorbar(cont)  # plot color bar
                fontC = {'family': 'Times New Roman', 'weight': 'normal', 'size': 35, }  # colorbar label size
                axisfont = 25  # axis number size
                cbar.ax.set_ylabel(y_label, fontC)  # color bar label
                cbar.ax.tick_params(labelsize=axisfont)  # color bar axis number size
                # pass
            else:  # plot color bar according to given range
                t=np.arange(y_range[0], y_range[1] + y_range[2] / 2, y_range[2])
                cbar = plt.colorbar(cont,ticks=t.tolist())  # plot color bar #bug: np.arange(15,20.1,0.1)
                fontC = {'family': 'Times New Roman', 'weight': 'normal', 'size': 35, } # colorbar label size
                axisfont = 25  # axis number size
                cbar.ax.set_ylabel(y_label, fontC)  # color bar label
                cbar.ax.tick_params(labelsize=axisfont)  # color bar axis number size

            # font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
            # plt.legend(handles=[p1], labels=['training samples'], loc='best', edgecolor='black', prop=font)

        if show_flag == 1:
            plt.subplots_adjust(top=0.85, bottom=0.05, right=0.9, left=0.2) # adjust side blank size
            # plt.show()
            plt.savefig('plot/surrogate_model_'+str(outer_iter)+'.eps')
            plt.close()

        if sample_flag == 1:
            return plt, p1
        else:
            return plt


    def plot_md(self, x1_arg, x2_arg, lower_bound, upper_bound, y_range=[], rest_var=None, show_flag_md=1, sample_flag=1):
        """
        plot multi dimension response surface of the selected two variables: x1_arg and x2_arg
        if rest_var==None, the rest variables are set as the mid point of lower_bound and upper_bound by default

        e.g. x1_arg=0, x2_arg=3, it will plot the response surface with the first and the fourth parameters as variables
        """
        plotgrid = 100 # the number of grid points of each varible, so the total number of grid points is plotgrid^2
        X = np.ones((plotgrid*plotgrid, len(lower_bound))) # initialize X
        try: # note: if rest_var!=None, it will report error, so we use try, and the function is the same
            if rest_var==None:
                rest_var=(np.array(lower_bound)+np.array(upper_bound))/2 # all variables are set as the mid point of lower_bound and upper_bound by default
        except: pass
        X=rest_var*X

        # set the selected two variables
        x1 = np.linspace(lower_bound[x1_arg], upper_bound[x1_arg], plotgrid)
        x2 = np.linspace(lower_bound[x2_arg], upper_bound[x2_arg], plotgrid)
        X1, X2 = np.meshgrid(x1, x2)

        # resize meshgrid to one column as the input for function
        X1.resize((X1.shape[0] * X1.shape[1]))
        X2.resize((X2.shape[0] * X2.shape[1]))
        X[:, x1_arg] = X1
        X[:, x2_arg] = X2

        y = self.calculate(X)  # predict
        y.resize((plotgrid, plotgrid))  # resize to meshgrid for plot

        # plot
        X1, X2 = np.meshgrid(x1, x2)

        colorslist = ['blue', 'aqua', 'yellow', 'red']  # color of color bar
        cmaps = colors.LinearSegmentedColormap.from_list('mylist', colorslist, N=3000)

        # fig = plt.figure(figsize=(10, 7.5))  # figure size

        if y_range == []:  # plot contour according to y range
            cont = plt.contourf(X1, X2, y, 50, cmap=cmaps)
            # cbar = plt.colorbar(cont)  # plot color bar

        else:  # plot contour according to given range
            y[y < y_range[0]] = y_range[0]  # assgin any y, which is smaller than the given lower boundary, to lower boundary
            y[y > y_range[1]] = y_range[1]  # assgin any y, which is larger than the given upper boundary, to upper boundary
            lim = np.arange(y_range[0]-(y_range[1] - y_range[0]) / 50, y_range[1] + (y_range[1] - y_range[0]) / 50*2, (y_range[1] - y_range[0]) / 50)
            cont = plt.contourf(X1, X2, y, lim, cmap=cmaps)  # plot contour
            # cbar = plt.colorbar(cont,ticks=np.arange(y_range[0], y_range[1] + y_range[2]/2, y_range[2]).tolist())  # plot color bar

        # plot scatter of all samples X
        if sample_flag == 1:
            p1 = plt.scatter(self.X[:, x1_arg], self.X[:, x2_arg], s=5, marker='o', c='black')  # scatter of all samples X
            plt.xlim(lower_bound[x1_arg], upper_bound[x1_arg])  # x range
            plt.ylim(lower_bound[x2_arg], upper_bound[x2_arg])  # x range
            # font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
            # plt.legend(handles=[p1], labels=['training samples'], loc='best', edgecolor='black', prop=font)

            if show_flag_md == 1: plt.show()
            return cont,p1

        else:
            if show_flag_md == 1: plt.show()
            return cont,None



if __name__ == '__main__':
    # ==================================================
    rootpath = r'D:\ljj\aa\demo'  # your saopy file path
    import sys

    sys.path.append(rootpath)  # you can directly import the modules in this folder
    sys.path.append(rootpath + r'\saopy\surrogate_model')
    sys.path.append(rootpath + r'\saopy')
    # ==================================================
    from saopy.surrogate_model.ANN import *
    # from saopy.surrogate_model.KRG import *
    # from saopy.surrogate_model.RBF import *
    from saopy.surrogate_model.surrogate_model import *

    dimension = 5
    lower_bound = [0] * dimension
    upper_bound = [1] * dimension
    # plot_y_range = [0,54,10]
    plot_y_range=[]
    x_label=['x1','x2','x3','x4','x5']

    best_surro=load_obj('best_surro0')
    best_surro.plot(lower_bound, upper_bound, plot_y_range, x_label=x_label,outer_iter=0, sample_flag=0)  # plot response surface
    # print(best_surro.calculate(np.array([[0.1,0.2,0.3,0.4,0.5,0.6,0.7]]))) # calculate a certain point