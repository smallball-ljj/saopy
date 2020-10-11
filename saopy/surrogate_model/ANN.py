# ==================================================
# original author: Daniel Guo
# https://github.com/d-guo/GenNet/blob/master/nn.py
# for more information about pytorch, visit https://pytorch.org/
# ==================================================
# modified by luojiajie
# ==================================================


import torch
import torch.nn as nn
import matplotlib.pyplot as plt
# import torch.nn.functional as F

from surrogate_model import *
import cross_validation
import csv
import numpy as np
import os


"""
old definition example

class FCNN(nn.Module):
    def __init__(self,input_shape):
        super().__init__()
        self.fc1=nn.Linear(input_shape, 100)
        self.fc2=nn.Linear(100, 100)
        self.fc3 = nn.Linear(100, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
"""

class FCNN(nn.Module):
    """
    definition of fully connected neural network
    """
    def __init__(self,input_shape, num_layers, neurons, activator_id=5, optimizer_id=0):
        super().__init__()
        self.num_layers = num_layers      # number of layers
        self.neurons = neurons            # number of neurons in each layer e.g. for 2 layers, neurons=[10,20]
        self.activator_id = activator_id  # activation function, can be one of the following: ELU, Hardshrink, LeakyReLU, LogSigmoid, PReLU, ReLU, ReLU6, RReLU, SELU, CELU, Sigmoid
        self.optimizer_id = optimizer_id  # optimizer id, can be one of the following: Adadelta, Adagrad, Adam, Adamax, ASGD, RMSprop, Rprop, SGD

        # set activation function
        if (activator_id == 0):
            self.activator = nn.ELU()
        elif (activator_id == 1):
            self.activator = nn.Hardshrink()
        elif (activator_id == 2):
            self.activator = nn.LeakyReLU()
        elif (activator_id == 3):
            self.activator = nn.LogSigmoid()
        elif (activator_id == 4):
            self.activator = nn.PReLU()
        elif (activator_id == 5):
            self.activator = nn.ReLU()
        elif (activator_id == 6):
            self.activator = nn.ReLU6()
        elif (activator_id == 7):
            self.activator = nn.RReLU()
        elif (activator_id == 8):
            self.activator = nn.SELU()
        elif (activator_id == 9):
            self.activator = nn.CELU()

        # network architecture
        if (num_layers == 1):
            self.layers = nn.Sequential(
                nn.Linear(input_shape, self.neurons[0]),
                self.activator,
                nn.Linear(self.neurons[0], 1)
            )
        elif (num_layers == 2):
            self.layers = nn.Sequential(
                nn.Linear(input_shape, self.neurons[0]),
                self.activator,
                nn.Linear(self.neurons[0], self.neurons[1]),
                self.activator,
                nn.Linear(self.neurons[1], 1)
            )
        elif (num_layers == 3):
            self.layers = nn.Sequential(
                nn.Linear(input_shape, self.neurons[0]),
                self.activator,
                nn.Linear(self.neurons[0], self.neurons[1]),
                self.activator,
                nn.Linear(self.neurons[1], self.neurons[2]),
                self.activator,
                nn.Linear(self.neurons[2], 1)
            )
        elif (num_layers == 4):
            self.layers = nn.Sequential(
                nn.Linear(input_shape, self.neurons[0]),
                self.activator,
                nn.Linear(self.neurons[0], self.neurons[1]),
                self.activator,
                nn.Linear(self.neurons[1], self.neurons[2]),
                self.activator,
                nn.Linear(self.neurons[2], self.neurons[3]),
                self.activator,
                nn.Linear(self.neurons[3], 1)
            )

    def forward(self, x):
        return self.layers(x)


class ANN(surrogate_model):
    def __init__(self, num_layers=2, num_neurons=100):
        super().__init__()
        # ==================================================
        # for simplicity, in this version, each layer has the same number of neurons
        self.num_layers = num_layers
        self.neurons = []
        for i in range(num_layers):
            self.neurons.append(num_neurons)
        # ==================================================


    def train(self, X_train, y_train, plot_train_history=0, epoch=10000):
        '''
        :param X_train,y_train: normalized training data
        :param plot_train_history: whether or not plot tran history. plot_train_history=0: not plot; plot_train_history=1:plot
        :param epoch: total training epoch. the larger the epoch, the longer training time. user should judge the value by the convergence of plot_train_history
        '''
        self.fcnn = FCNN(input_shape=X_train.shape[1], num_layers=self.num_layers, neurons=self.neurons)  # instantiate FCNN

        loss_criterion=nn.MSELoss() # use mean squared error loss
        if (self.fcnn.optimizer_id == 0): # for comparison of different optimizer, visit: https://blog.csdn.net/qq_35082030/article/details/73368962
            optimizer = torch.optim.Adadelta(self.fcnn.parameters())

        loss_history=[]
        for iter in range(0, epoch):
            optimizer.zero_grad() # zero the gradient, or it will be accumulated
            self.y_pred = self.fcnn(torch.from_numpy(X_train).float()) # forward propagation
            loss = loss_criterion(self.y_pred, torch.from_numpy(y_train).float()) # calculate loss
            loss.backward() # backward propagation
            optimizer.step() # update weight

            loss_history.append(loss.detach().numpy()) # record loss

            # if (iter % 100 == 0):
            #     print(loss)

        if plot_train_history==1:
            if os.path.exists('plot') == False:
                os.makedirs('plot')
            plt.plot(range(1,epoch+1),loss_history)
            plt.xlabel('epoch')
            plt.ylabel('MSE of training data')
            plt.savefig('plot/ANN_train_history.png')
            plt.close()


    def calculate(self, X):
        """
        :param X: numpy array, with shape(number,dimension)
        """
        X=self.normalize_X(X)
        X=torch.from_numpy(X).float() # convert numpy to tensor
        y=self.fcnn(X)
        y=y.detach().numpy() # convert tensor to numpy
        y=self.inverse_normalize_y(y)
        return y




def get_best_arch(lower_bound,upper_bound,file_X,file_y,max_layers=3,max_neurons=100,step=1,num_fold=3,parallel_num=1):
    """
    get best ANN architecture with minimum RMSE
    the total number of ANN architecture for testing is max_layers*max_neurons

    :param lower_bound: lower boundary of X
    :param upper_bound: upper boundary of X
    :param file_X: X.csv
    :param file_y: y.csv
    :param max_layers: max layers of ANN
    :param max_neurons: max neurons of ANN
    :param step: step of neurons
    :param num_fold: fold number for cross validation
    :param parallel_num: parallel process number for cross validation
    :return: best ANN with minimum RMSE
    """
    surro_list=[]
    for num_layers in range(1, max_layers+1):
        for num_neurons in range(1, max_neurons+1, step):
            surro_list.append(ANN(num_layers=num_layers, num_neurons=num_neurons)) # get all architectures of ANN, you can also defined by yourself

    for surro in surro_list:
        surro.load_data(lower_bound, upper_bound, file_X, file_y)
        surro.normalize_all()
    cros_valid=cross_validation.random(surro_list,num_fold=num_fold)
    cros_valid.divide()
    best_ind = cros_valid.begin_cross_validation(parallel_num) # get best model index
    best_surro = surro_list[best_ind] # get best model with minimum RMSE
    print('best ANN architecture with min RMSE:','\n','num_layers=', best_surro.num_layers, ', num_neurons=', best_surro.neurons[0])
    return best_surro


def get_best_arch_plot_RMSE(max_layers,max_neurons,step,save_ind):
    marker_list=['s','^','o','v']
    color_list = ['blue','red','black','orange']
    label_list=[]
    plt.figure(figsize=(10, 8))

    RMSE_mean = read_csv_to_np('RMSE_mean.csv')
    np.savetxt('RMSE_mean_ANN_arch_'+str(save_ind)+'.csv', RMSE_mean, delimiter=',')
    for num_layers in range(max_layers):
        label_list.append('number of hidden layer = '+str(num_layers+1))
        node=(max_neurons-1)//step+1
        start_ind=num_layers*node
        end_ind=start_ind+node
        RMSE = RMSE_mean[start_ind:end_ind]
        plt.plot(range(1,max_neurons+1,step),RMSE,marker=marker_list[num_layers],c=color_list[num_layers])

    plt.tick_params(labelsize=40)  # axis number size
    font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 40, }
    font2 = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
    plt.xlabel('Number of neurons per layer', font)
    plt.ylabel('Average RMSE of all trials', font)
    plt.legend(labels=label_list, loc='best', edgecolor='black', prop=font2)
    # plt.ylim(0.08,0.22)
    plt.subplots_adjust(top=0.95, bottom=0.2, right=0.95, left=0.25)
    # plt.show()
    plt.savefig('RMSE_mean_ANN_arch_'+str(save_ind)+'.eps')
    plt.close()


def read_csv_to_np(file_name):
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



if __name__ == '__main__':
    # example for getting best ANN architecture with minimum RMSE
    # note: you must copy X.csv any y.csv to this folder
    # multiprocessing note: multiprocessing.Pool is applied to cross_validation and it must run after <if __name__ == '__main__':>

    lower_bound = [0, 0]
    upper_bound = [1, 1]

    max_layers = 2 # max layers of ANN
    max_neurons = 1 # max neurons of ANN
    step=10 # step of neurons
    num_fold=3 # fold number for cross validation
    parallel_num=3 # parallel process number for cross validation

    best_ANN_arch=get_best_arch(lower_bound,upper_bound,'X.csv','y.csv',max_layers,max_neurons,step,num_fold,parallel_num)
    get_best_arch_plot_RMSE(max_layers, max_neurons,step,0)
    save_obj(best_ANN_arch, 'best_ANN_arch')
    best_ANN_arch = load_obj('best_ANN_arch')
    best_num_layers = best_ANN_arch.num_layers
    best_num_neurons = best_ANN_arch.neurons[0]
