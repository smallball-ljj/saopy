# ==================================================
rootpath=r'F:\ljj\aa\demo' # your saopy file path
import sys
sys.path.append(rootpath) # you can directly import the modules in this folder
sys.path.append(rootpath+r'\saopy\surrogate_model')
sys.path.append(rootpath+r'\saopy')
# ==================================================

from saopy.sampling_plan import *
from saopy.function_evaluation.benchmark_func import *
from saopy.surrogate_model.ANN import *
from saopy.surrogate_model.KRG import *
from saopy.surrogate_model.RBF import *
from saopy.surrogate_model.surrogate_model import *
from saopy.surrogate_model import cross_validation
from saopy.optimization import *
from saopy.exploration import *
from saopy.exploitation import *
from saopy.adaptive_sequential_sampling import *
from saopy.database import *
from saopy.surro_valid_compare import *

import numpy as np
import time

# multiprocessing note: multiprocessing.Pool is applied to cross_validation and it must run after <if __name__ == '__main__':>
# multiprocessing note: you will not get the class attribute if you run the method in other process in parallel
# e.g. run <self.point=1> in other process, you will not get the self.point in the main process

if __name__ == '__main__':

    lower_bound = [-32.768, -32.768]
    upper_bound = [32.768, 32.768]
    plot_y_range = [0, 20, 2]

    total_iter=20

    for iter in range(total_iter):
        print(iter)
        if iter==0:
            print('initial sampling plan')
            number = 40; dimension = 2
            sp = optimal_lhs(number, dimension)
            sp.begin_sampling(population=30, iterations=30)  # the larger the better, but will take long time
            sp.inverse_norm(lower_bound, upper_bound)
            sp.output('X_new.csv')
            # sp.plot(lower_bound, upper_bound)
        print('function evaluation')
        f = ackley(2)
        X = f.read_csv_to_np('X_new.csv')
        f.calculate(X)
        f.output('y0_new.csv')
        f.plot(lower_bound, upper_bound, plot_y_range)
        print('database')
        stack('X.csv','X_new.csv')
        stack('y0.csv','y0_new.csv')
        if iter==0:
            print('get best ANN architecture with minimum RMSE')
            max_layers = 3
            max_neurons = 101
            step = 50
            num_fold = 3
            parallel_num = 30
            best_ANN_arch = get_best_arch(lower_bound, upper_bound, 'X.csv', 'y0.csv', max_layers, max_neurons, step, num_fold, parallel_num)
            get_best_arch_plot_RMSE(max_layers, max_neurons, step, 0)
            save_obj(best_ANN_arch, 'best_ANN_arch')

        print('cross validation and get max error point')
        parallel_num = 3; num_fold = 3
        best_ANN_arch = load_obj('best_ANN_arch')
        best_num_layers = best_ANN_arch.num_layers
        best_num_neurons = best_ANN_arch.neurons[0]
        surro_list=[]
        surro_list.append(ANN(num_layers=best_num_layers, num_neurons=best_num_neurons))
        # surro_list.append(RBF(num_centers=10))
        # surro_list.append(KRG()) # may take longer training time
        for surro in surro_list:
            surro.load_data(lower_bound, upper_bound,'X.csv','y0.csv')
            surro.normalize_all()
        cros_valid=cross_validation.random(surro_list,num_fold=num_fold)
        cros_valid.divide()
        best_ind = cros_valid.begin_cross_validation(parallel_num,plot_flag=1) # get best model index
        best_surro = surro_list[best_ind] # get best model
        best_surro.train(best_surro.normalized_X,best_surro.normalized_y) # train again using all samples
        save_obj(best_surro,'best_surro') # save best model
        best_surro.plot(lower_bound, upper_bound,plot_y_range,outer_iter=iter) # plot response surface

        print('adaptive sequential sampling')
        surro_list = []
        surro_list.append(load_obj('best_surro'))
        ass1 = ass(surro_list, iter, plot_flag=0)
        ass1.generate_new_X(number_optimized_X=5, number_exploitation_X=2, number_exploration_X=2)
        ass1.plot(surro_list[0],plot_y_range,iter)
