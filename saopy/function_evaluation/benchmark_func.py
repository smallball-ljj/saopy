# ==================================================
# author:luojiajie
# ==================================================

import os
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors


class benchmark_func():
    """
    base class for all benchmark functions
    """
    def __init__(self,dimension=None):
        self.d = dimension # dimension of function if available
        self.y = None # calculated function value


    def calculate(self,X):
        """
        detailed function definition
        :param  X: numpy array, with shape(number,dimension)
        :return y: numpy array, with shape(number,1)
        """
        pass


    def read_csv_to_np(self,file_name='X.csv'):
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


    def output(self,file_name='y.csv'):
        """
        output self.y to <file_name>
        """
        np.savetxt(file_name, self.y, delimiter=',')


    def plot(self, lower_bound, upper_bound, y_range=[], show_flag=1):
        """
        plot function

        :param lower_bound: lower boundary of X   e.g. for 2D: lower_bound=[-1,-2], for 1D: lower_bound=[-1]
        :param upper_bound: upper boundary of X   e.g. for 2D: upper_bound=[3,4], for 1D: upper_bound=[3]
        :param y_range: 1. if y_range==[], plot y according to calculated max range of y
                        2. plot y acoording to given range: y_range=[lower boundary, upper boundary, step]
                        e.g. y_range=[0,10,1]
        :param show_flag: if show_flag==1, show the plot (default)
                          if show_flag==0, do not show the plot (which will return plt, for other class to call it)
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
            p=plt.plot(X, y, linestyle='-', linewidth=3, c='black',label='benchmark function')
            plt.xlim(lower_bound[0], upper_bound[0]) # x range
            if y_range==[]:
                pass
            else:
                plt.ylim(y_range[0], y_range[1]) # y range
            # legend
            font_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
            plt.legend(loc='best',edgecolor='black', prop=font_legend)
            # xy label size
            font_xy = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
            plt.xlabel('x', font_xy)
            plt.ylabel('y', font_xy)
            # coordinate font
            plt.tick_params(labelsize=20)

        # ==================================================
        # multi dimension
        # ==================================================
        elif dimension >= 2:
            fig = plt.figure(figsize=(10, 7.5))  # figure size
            for i in range(dimension):
                for j in range(dimension-1, i, -1):
                    col_ind = i # column index
                    row_ind = -j + dimension - 1 # row index
                    ind = row_ind * (dimension - 1) + col_ind + 1 # subplot index
                    plt.subplot(dimension - 1, dimension - 1, ind) # locate subplot
                    cont=self.plot_md(i, j, lower_bound, upper_bound, y_range, show_flag_md=0) # plot response surface in subplot
                    # plot axis label
                    fontXY = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }  # xy label size
                    axisfont = 20  # axis number size
                    if row_ind == 0 and col_ind == 0:  # first subplot, plot both axis
                        ax = plt.gca()
                        ax.xaxis.set_ticks_position('top')
                        ax.xaxis.set_label_position('top')
                        plt.xlabel('x' + str(i+1), fontXY)
                        plt.ylabel('x' + str(j+1), fontXY)
                    elif row_ind == 0 and col_ind != 0:  # first row, plot x axis label
                        plt.yticks([])
                        ax = plt.gca()
                        ax.xaxis.set_ticks_position('top')
                        ax.xaxis.set_label_position('top')
                        plt.xlabel('x' + str(i+1), fontXY)
                    elif col_ind == 0 and row_ind != 0:  # first column, plot y axis label
                        plt.xticks([])
                        plt.ylabel('x' + str(j+1), fontXY)
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
                # cbar = plt.colorbar(cont)  # plot color bar
                pass
            else:  # plot color bar according to given range
                t = np.arange(y_range[0], y_range[1] + y_range[2] / 2, y_range[2])
                cbar = plt.colorbar(cont, ticks=t.tolist())  # plot color bar #bug: np.arange(15,20.1,0.1)
                fontC = {'family': 'Times New Roman', 'weight': 'normal', 'size': 25, }  # colorbar label size
                axisfont = 20  # axis number size
                cbar.ax.set_ylabel('y', fontC)  # color bar label
                cbar.ax.tick_params(labelsize=axisfont)  # color bar axis number size

            # font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
            # plt.legend(handles=[p1], labels=['training samples'], loc='best', edgecolor='black', prop=font)

        if show_flag == 1:
            # plt.show()
            plt.savefig('plot/benchmark_func.eps')
            plt.close()
        return plt


    def plot_md(self, x1_arg, x2_arg, lower_bound, upper_bound, y_range=[], show_flag_md=1):
        """
        plot multi dimension response surface of the selected two variables: x1_arg and x2_arg
        the rest variables are set as the mid point of lower_bound and upper_bound by default

        e.g. x1_arg=0, x2_arg=3, it will plot the response surface with the first and the fourth parameters as variables
        """
        plotgrid = 100 # the number of grid points of each varible, so the total number of grid points is plotgrid^2
        X = np.ones((plotgrid*plotgrid, len(lower_bound))) # initialize X
        rest_var=(np.array(lower_bound)+np.array(upper_bound))/2 # all variables are set as the mid point of lower_bound and upper_bound by default
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
            cbar = plt.colorbar(cont)  # plot color bar

        else:  # plot contour according to given range
            y[y < y_range[0]] = y_range[0]  # assgin any y, which is smaller than the given lower boundary, to lower boundary
            y[y > y_range[1]] = y_range[1]  # assgin any y, which is larger than the given upper boundary, to upper boundary
            lim = np.arange(y_range[0]-(y_range[1] - y_range[0]) / 50, y_range[1] + (y_range[1] - y_range[0]) / 50*2, (y_range[1] - y_range[0]) / 50)
            cont = plt.contourf(X1, X2, y, lim, cmap=cmaps)  # plot contour
            # cbar = plt.colorbar(cont,ticks=np.arange(y_range[0], y_range[1] + y_range[2]/2, y_range[2]).tolist())  # plot color bar

        if show_flag_md == 1: plt.show()
        return cont




# ==================================================
# detailed definition of benchmark functions (used for single objective optimization)
# reference:
# X.-S. Yang, Test problems in optimization, in: Engineering Optimization: An Introduction with Metaheuristic Applications (Eds Xin-She Yang), John Wiley & Sons, (2010).
# ==================================================
class sin(benchmark_func):
    """
    dimension=1
    x range: [-1,1]
    """
    def calculate(self,X):
        self.y=np.sin(2 * np.pi * X)
        self.y.resize((self.y.shape[0], 1))
        return self.y

class d1f1(benchmark_func):
    """
    dimension=1
    x range: [0,1]
    global minimum: f(0.7572)=-6.020
    """
    def calculate(self, X):
        self.y = (6*X-2)**2*np.sin(2*(6*X-2))
        self.y.resize((self.y.shape[0], 1))
        return self.y

class ackley(benchmark_func):
    """
    dimension>=1
    x range: [-32.768,32.768]
    global minimum: f(0,...,0)=0
    """
    def calculate(self, X):
        a = -20 * np.exp(-0.2 * np.sqrt(1 / self.d * np.sum(X ** 2, axis=1)))
        b = -np.exp(1 / self.d * np.sum(np.cos(2 * np.pi * X), axis=1))
        self.y = 20 + np.e + a + b
        self.y.resize((self.y.shape[0], 1))
        return self.y

class sphere(benchmark_func):
    """
    dimension>=1
    x range: [-5.12,5.12]
    global minimum: f(0,...,0)=0
    """
    def calculate(self, X):
        self.y = np.sum(X ** 2, axis=1)
        self.y.resize((self.y.shape[0],1))
        return self.y

class sphere_weighted(benchmark_func):
    """
    dimension>=1
    x range: [-5.12,5.12]
    global minimum: f(0,...,0)=0
    """
    def calculate(self, X):
        co = np.arange(1, self.d+1)
        self.y = np.sum(co * X ** 2, axis=1)
        self.y.resize((self.y.shape[0],1))
        return self.y

class sphere_powered(benchmark_func):
    """
    dimension>=1
    x range: [-1,1]
    global minimum: f(0,...,0)=0
    """
    def calculate(self, X):
        self.y = np.zeros((X.shape[0],1))
        for i in range(self.d):
            self.y += np.abs(X[:, [i]])**(i+2)
        self.y.resize((self.y.shape[0],1))
        return self.y

class easom(benchmark_func):
    """
    dimension>=1
    x range: [-2*np.pi,2*np.pi]
    global minimum: f(np.pi,...,np.pi)=-1
    """
    def calculate(self, X):
        a = -np.sum((X-np.pi) ** 2, axis=1)
        b = -np.prod(np.cos(X) ** 2, axis=1)
        self.y = b * np.exp(a)
        self.y.resize((self.y.shape[0],1))
        return self.y

class griewank(benchmark_func):
    """
    dimension>=1
    x range: [-600,600]
    global minimum: f(0,...,0)=0
    """
    def calculate(self, X):
        a = 1/4000*np.sum(X ** 2, axis=1)
        co = np.arange(1, self.d + 1)**(-0.5)
        b = -np.prod(np.cos(X * co), axis=1)
        self.y = a+b+1
        self.y.resize((self.y.shape[0],1))
        return self.y

class michaelwicz(benchmark_func):
    """
    dimension>=1
    x range: [0,np.pi]
    global minimum: for 2D case: f(2.20319, 1.57049)= −1.8013
    """
    def calculate(self, X):
        co = np.arange(1, self.d + 1)
        self.y = -np.sum(np.sin(X)*(np.sin(co/np.pi*(X**2))**20), axis=1)
        self.y.resize((self.y.shape[0],1))
        return self.y

class perm(benchmark_func):
    """
    dimension>=1
    x range: [-1,1]
    global minimum: f(1,1/2,1/3...,1/n)=0
    """
    def calculate(self, X):
        b=100 # b is defined by user, should >0
        self.y = np.zeros((X.shape[0], 1))
        for j in range(1,self.d+1):
            y_tmp = np.zeros((X.shape[0], 1))
            for i in range(1,self.d+1):
                y_tmp += (i+b)*(X[:,[i-1]]**j-(1/i)**j)
            self.y += y_tmp**2
        self.y.resize((self.y.shape[0], 1))
        return self.y

class rastrigin(benchmark_func):
    """
    dimension>=1
    x range: [-5.12,5.12]
    global minimum: f(0,...,0)=0
    """
    def calculate(self, X):
        self.y = 10*self.d + np.sum(X**2-10*np.cos(2*np.pi*X), axis=1)
        self.y.resize((self.y.shape[0], 1))
        return self.y

class rosenbrock(benchmark_func):
    """
    dimension>=2
    x range: [-5,5]
    global minimum: f(1,...,1)=0
    """
    def calculate(self, X):
        self.y = np.zeros((X.shape[0],1))
        for i in range(self.d-1):
            self.y += 100 * ((X[:, [i+1]] - X[:, [i]] ** 2) ** 2)+(1 - X[:, [i]]) ** 2
        self.y.resize((self.y.shape[0], 1))
        return self.y

class schwefel(benchmark_func):
    """
    dimension>=1
    x range: [-500,500]
    global minimum: f(420.9687,...,420.9687)=−418.9829n
    """
    def calculate(self, X):
        self.y = -np.sum(X*np.sin(np.abs(X)**0.5), axis=1)
        self.y.resize((self.y.shape[0], 1))
        return self.y

class shubert(benchmark_func):
    """
    dimension=1
    x range: [-10,10]
    global minimum: f(?,?)=−186.7309
    """
    def calculate(self, X):
        a = np.zeros((X.shape[0],1))
        b = np.zeros((X.shape[0], 1))
        for i in range(1,6):
            a += i*np.cos(i+(i+1)*X[:, [0]])
        for i in range(1,6):
            b += i*np.cos(i+(i+1)*X[:, [1]])
        self.y = a*b
        self.y.resize((self.y.shape[0], 1))
        return self.y

class test1(benchmark_func):
    """
    dimension=1
    x range: [-1,1]
    """
    def calculate(self,X):
        self.y=X[:,0]**2+0.5*X[:,1]
        self.y.resize((self.y.shape[0], 1))
        return self.y

# ==================================================
# detailed definition of benchmark functions (used for multi objective optimization)
# ==================================================
class ZDT1_obj0(benchmark_func):
    """
    dimension=30
    x range: [0,1]
    """
    def calculate(self, X):
        self.y=X[:,[0]]
        self.y.resize((self.y.shape[0], 1))
        return self.y
class ZDT1_obj1(benchmark_func):
    def calculate(self, X):
        gx = 1 + 9/29 * np.sum(X[:, 1:30], axis=1)
        hx = 1 - (X[:, 0] / gx) ** 0.5
        self.y = gx * hx
        self.y.resize((self.y.shape[0], 1))
        return self.y

class ZDT1_m_obj0(benchmark_func):
    """
    dimension=2
    x range: [0,1]
    """
    def calculate(self, X):
        self.y=X[:,[0]]
        self.y.resize((self.y.shape[0], 1))
        return self.y
class ZDT1_m_obj1(benchmark_func):
    def calculate(self, X):
        gx = 1 + np.sum(X[:, 1:2], axis=1)
        hx = 1 - (X[:, 0] / gx) ** 0.5
        self.y = gx * hx
        self.y.resize((self.y.shape[0], 1))
        return self.y

class ZDT2_obj0(benchmark_func):
    """
    dimension=30
    x range: [0,1]
    """
    def calculate(self, X):
        self.y=X[:,[0]]
        self.y.resize((self.y.shape[0], 1))
        return self.y
class ZDT2_obj1(benchmark_func):
    def calculate(self, X):
        gx = 1 + 9/29 * np.sum(X[:, 1:30], axis=1)
        hx = 1 - (X[:, 0] / gx) ** 2
        self.y = gx * hx
        self.y.resize((self.y.shape[0], 1))
        return self.y

class ZDT3_obj0(benchmark_func):
    """
    dimension=30
    x range: [0,1]
    """
    def calculate(self, X):
        self.y=X[:,[0]]
        self.y.resize((self.y.shape[0], 1))
        return self.y
class ZDT3_obj1(benchmark_func):
    def calculate(self, X):
        gx = 1 + 9/29 * np.sum(X[:, 1:30], axis=1)
        hx = 1 - (X[:, 0] / gx) ** 0.5 - (X[:, 0] / gx) * np.sin(10 * np.pi * X[:, 0])
        self.y = gx * hx
        self.y.resize((self.y.shape[0], 1))
        return self.y

class DTLZ1_obj0(benchmark_func):
    """
    dimension>=2
    x range: [0,1]
    """
    def calculate(self, X):
        M = 3
        XM = X[:, (M - 1):].copy()
        g = 100 * (XM.shape[1] + np.sum(((XM - 0.5) ** 2 - np.cos(20 * np.pi * (XM - 0.5))), 1, keepdims=True))
        self.y = 0.5*X[:,[0]]*X[:,[1]]*(1+g)
        self.y.resize((self.y.shape[0], 1))
        return self.y
class DTLZ1_obj1(benchmark_func):
    def calculate(self, X):
        M = 3
        XM = X[:, (M - 1):].copy()
        g = 100 * (XM.shape[1] + np.sum(((XM - 0.5) ** 2 - np.cos(20 * np.pi * (XM - 0.5))), 1, keepdims=True))
        self.y = 0.5*X[:,[0]]*(1-X[:,[1]])*(1+g)
        self.y.resize((self.y.shape[0], 1))
        return self.y
class DTLZ1_obj2(benchmark_func):
    def calculate(self, X):
        M = 3
        XM = X[:, (M - 1):].copy()
        g = 100 * (XM.shape[1] + np.sum(((XM - 0.5) ** 2 - np.cos(20 * np.pi * (XM - 0.5))), 1, keepdims=True))
        self.y = 0.5*(1-X[:,[0]])*(1+g)
        self.y.resize((self.y.shape[0], 1))
        return self.y

class DTLZ1_m_obj0(benchmark_func):
    """
    dimension=2
    x range: [0,1]
    """
    def calculate(self, X):
        self.y = 0.5*X[:,[0]]*X[:,[1]]
        self.y.resize((self.y.shape[0], 1))
        return self.y
class DTLZ1_m_obj1(benchmark_func):
    def calculate(self, X):
        self.y = 0.5*X[:,[0]]*(1-X[:,[1]])
        self.y.resize((self.y.shape[0], 1))
        return self.y
class DTLZ1_m_obj2(benchmark_func):
    def calculate(self, X):
        self.y = 0.5*(1-X[:,[0]])
        self.y.resize((self.y.shape[0], 1))
        return self.y

class DTLZ1_m2_obj0(benchmark_func):
    """
    dimension>=2
    x range: [0,1]

    note: reduced cosine part in DTLZ1, which is simpler and don't has many local minimum
    """
    def calculate(self, X):
        M = 3
        XM = X[:, (M - 1):].copy()
        g = 100 * (XM.shape[1] + np.sum(((XM - 0.5) ** 2), 1, keepdims=True))
        self.y = 0.5*X[:,[0]]*X[:,[1]]*(1+g)
        self.y.resize((self.y.shape[0], 1))
        return self.y
class DTLZ1_m2_obj1(benchmark_func):
    def calculate(self, X):
        M = 3
        XM = X[:, (M - 1):].copy()
        g = 100 * (XM.shape[1] + np.sum(((XM - 0.5) ** 2), 1, keepdims=True))
        self.y = 0.5*X[:,[0]]*(1-X[:,[1]])*(1+g)
        self.y.resize((self.y.shape[0], 1))
        return self.y
class DTLZ1_m2_obj2(benchmark_func):
    def calculate(self, X):
        M = 3
        XM = X[:, (M - 1):].copy()
        g = 100 * (XM.shape[1] + np.sum(((XM - 0.5) ** 2), 1, keepdims=True))
        self.y = 0.5*(1-X[:,[0]])*(1+g)
        self.y.resize((self.y.shape[0], 1))
        return self.y


# e.g.
if __name__ == '__main__':
    dimension=5

    # lower_bound = [-32.768]*dimension # ackley
    # upper_bound = [32.768]*dimension

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

    lower_bound = [0]*dimension # shubert
    upper_bound = [1]*dimension

    # f = ackley(dimension)
    # f = sphere(dimension)
    # f = sphere_weighted(dimension)
    # f = sphere_powered(dimension)
    # f = easom(dimension)
    # f = griewank(dimension)
    # f = michaelwicz(dimension)
    # f = perm(dimension)
    # f = rastrigin(dimension)
    # f = rosenbrock(dimension)
    # f = schwefel(dimension)
    # f = shubert(dimension)
    f = DTLZ1_m2_obj0(dimension)


    # X = f.read_csv_to_np('output_sampling_plan.csv')
    # f.calculate(X)
    # f.output('output_func.csv')
    f.plot(lower_bound, upper_bound,[0,80,10])