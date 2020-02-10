# ==================================================
# author:luojiajie
# ==================================================
"""
useage:
    divide self.normalized_X and self.normalized_y of a list of surrogate models into the same n fold. then, use n-1 fold as training data and 1 fold as testing data for cross validation
    note: self.normalized_X and self.normalized_y in the list of surrogate models are divided into the same n fold, so that the results of cross validation of different models are comparable

Example:
    suppose:
    surro.normalized_X=
    [[0.25 0.85]
     [0.95 0.15]
     [0.05 0.45]
     [0.45 0.65]
     [0.65 0.95]
     [0.55 0.75]
     [0.85 0.55]
     [0.35 0.35]
     [0.15 0.05]
     [0.75 0.25]]

    run:
    div=random([surro1,surro2],num_fold=3)
    div.divide()

    then:
    rand_seq=[4 6 5 7 9 1 3 0 8 2]
    surro.normalized_fold_X=
    [array([[0.65, 0.95],
           [0.85, 0.55],
           [0.55, 0.75]]), array([[0.35, 0.35],
           [0.75, 0.25],
           [0.95, 0.15]]), array([[0.45, 0.65],
           [0.25, 0.85],
           [0.15, 0.05],
           [0.05, 0.45]])]
"""
import os
import csv
import numpy as np
from multiprocessing import Pool
import matplotlib.pyplot as plt


class cross_validation():
    def __init__(self,surro_list,num_fold=3):
        """
        :param surro_list: list of surrogate model object (which has already run surro.load_data(), surro.normalize_all(), so that surro.normalized_X is got)
        :param num_fold: number of fold
        """
        self.surro_list=surro_list
        self.num_fold=num_fold
        self.samples_per_fold = self.surro_list[0].normalized_X.shape[0] // self.num_fold  # rounding down. samples per fold (except for the last fold, if not divisible)


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


    def divide_surro_0(self):
        """
        detailed divide method defined in the specific class
        note: this method will divide the first surrogate model in self.surro_list (surro.normalized_fold_X, surro.normalized_fold_y are got)
        so when run divide(), this method will be called first,
        then, surro.normalized_fold_X and surro.normalized_fold_y of the rest of surrogate models in self.surro_list will be assigned as the same value
        """
        pass

    def divide(self):
        """
        divide the first surrogate model, then assigned the rest as the same value
        """
        self.divide_surro_0() # divide the first surrogate model

        for i in range(1,len(self.surro_list)): # assigned the rest as the same value
            self.surro_list[i].normalized_fold_X = self.surro_list[0].normalized_fold_X
            self.surro_list[i].normalized_fold_y = self.surro_list[0].normalized_fold_y


    def begin_cross_validation(self, parallel_num, plot_flag=0, obj_ind=0):
        """
        cross validation all surrogate model in self.surro_list

        :param parallel_num: total parallel process number (must >=1 )
        :param plot_flag: plot Predicted Value vs Ground Truth
        :param obj_ind: objective index for saving RMSE_mean_min.csv (use for multi objective optimization)
        output: mean RMSE of all fold of all surrogate model in self.surro_list
        """
        if os.path.exists('plot') == False:
            os.makedirs('plot')

        pool=Pool(processes=parallel_num) # create processes

        parallel_process = [] # parallel process list
        for surro in self.surro_list: # append parallel process
            self.append_parallel_process(surro, parallel_process, pool, plot_flag)
        # pool.close() #
        # pool.join() # wait all the process to finish. this has bug when running on HPC, maybe due to the version, has not fixed yet

        RMSE_mean = [] # mean RMSE of all fold of each surrogate model
        RMSE = np.zeros((self.num_fold, 1)) # initialize RMSE list of all fold
        max_test_error = np.zeros((self.num_fold, 1)) # initialize max_test_error list of all fold
        point_max_error = np.zeros((self.num_fold, self.surro_list[0].X.shape[1])) # initialize point_max_error list of all fold
        fold_ind = 0 # fold index
        surro_ind = 0 # surrogate model index
        for p in parallel_process:
            RMSE[fold_ind], max_test_error[fold_ind], point_max_error[fold_ind] = p.get() # run in parallel
            fold_ind += 1
            if fold_ind % self.num_fold == 0: # all fold cross validation of this surrogate model is completed
                self.surro_list[surro_ind].point_max_error = point_max_error[max_test_error.argmax()]  # get self.point_max_error for exploitation.py to call it
                self.surro_list[surro_ind].max_test_error = max_test_error.max() # get self.max_test_error for exploitation.py to call it
                RMSE_mean.append(RMSE.mean()) # get mean RMSE of all fold of this surrogate model, and append
                surro_ind += 1
                RMSE = np.zeros((self.num_fold, 1))  # initialize RMSE list of all fold
                max_test_error = np.zeros((self.num_fold, 1))  # initialize max_test_error list of all fold
                point_max_error = np.zeros((self.num_fold, self.surro_list[0].X.shape[1]))  # initialize point_max_error list of all fold
                fold_ind = 0  # fold index

        # save RMSE_mean
        RMSE_mean = np.array(RMSE_mean)
        RMSE_mean = RMSE_mean.T
        np.savetxt('RMSE_mean.csv', RMSE_mean, delimiter=',')

        # save RMSE_mean.min()
        if os.path.exists('RMSE_mean_min'+str(obj_ind)+'.csv') == False:
            np.savetxt('RMSE_mean_min'+str(obj_ind)+'.csv', np.array([RMSE_mean.min()]), delimiter=',')
        else:
            RMSE_mean_min_old = self.read_csv_to_np('RMSE_mean_min'+str(obj_ind)+'.csv')
            RMSE_mean_min = np.vstack((RMSE_mean_min_old, np.array([RMSE_mean.min()])))
            np.savetxt('RMSE_mean_min'+str(obj_ind)+'.csv', RMSE_mean_min, delimiter=',')

        pool.close() # comment this, if you use <pool.close()> and <pool.joint()> in line 114,115
        return RMSE_mean.argmin() # return surrogate model index with minimum RMSE_mean

    def append_parallel_process(self, surro, parallel_process, pool, plot_flag):
        """
        use n-1 fold as training data and 1 fold as testing data for single surrogate model in self.surro_list
        training and getting RMSE for each fold can run in parallel, so we append it to parallel_process

        :param surro: single surrogate model in self.surro_list
        :param parallel_process: list of parallel process
        :param pool: multiprocess class
        """
        for i in range(len(surro.normalized_fold_X)):
            X_test_normalized = surro.normalized_fold_X[i].copy()  # get 1 fold of testing data
            y_test_normalized = surro.normalized_fold_y[i].copy()

            X_train_empty_flag = 1  # whether X_train is empty flag (we use this because np.vstack cannot stack empty array)
            for j in range(len(surro.normalized_fold_X)):  # stack n-1 fold of training data to X_train
                if j == i: continue  # this fold has been assigned as testing data, skip
                if X_train_empty_flag:  # if X_train is empty, it will be directly assigned as the first fold
                    X_train_normalized = surro.normalized_fold_X[j].copy()
                    y_train_normalized = surro.normalized_fold_y[j].copy()
                    X_train_empty_flag = 0  # now X_train is not empty
                else:  # if X_train is not empty, the next fold will be stacked to it
                    X_train_normalized = np.vstack((X_train_normalized, surro.normalized_fold_X[j]))
                    y_train_normalized = np.vstack((y_train_normalized, surro.normalized_fold_y[j]))

            parallel_process.append(pool.apply_async(self.get_RMSE_and_point_max_error,(surro, X_train_normalized, y_train_normalized, X_test_normalized, y_test_normalized, plot_flag,i, )))  # append parallel process

    def get_RMSE_and_point_max_error(self, surro, X_train_normalized, y_train_normalized, X_test_normalized, y_test_normalized, plot_flag,i):
        """
        get RMSE and point_max_error of single fold
        """
        surro.train(X_train_normalized, y_train_normalized)  # train the surrogate model use n-1 fold training data
        X_test=surro.inverse_normalize_X(X_test_normalized)
        y_test_pred = surro.calculate(X_test)  # predict the y of testing data use the trained model
        y_test_pred_normalized=surro.normalize_y(y_test_pred)
        RMSE = pow(pow((y_test_pred_normalized - y_test_normalized), 2).mean(), 0.5)  # calculate RMSE of y of testing data

        # get the point of max error
        test_error = pow((y_test_pred_normalized - y_test_normalized), 2)  # calculate the error between true y and predicted y of testing data
        max_test_error = test_error.max()  # get max test error
        point_max_error = X_test_normalized[test_error.argmax()]  # get the point of max error

        if plot_flag==1: # plot Predicted Value vs Ground Truth
            X_train = surro.inverse_normalize_X(X_train_normalized)
            y_train_pred = surro.calculate(X_train)
            y_train = surro.inverse_normalize_y(y_train_normalized)
            y_test = surro.inverse_normalize_y(y_test_normalized)
            self.plot(y_train,y_train_pred,y_test,y_test_pred,surro,i)

        return RMSE, max_test_error, point_max_error


    def plot(self,y_train,y_train_pred,y_test,y_test_pred,surro,i):
        font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
        plt.figure(figsize=(10, 8))
        plt.xlabel('Ground Truth', font)
        plt.ylabel('Predicted Value', font)
        plt.xlim(surro.y_min, surro.y_max)
        plt.ylim(surro.y_min, surro.y_max)
        plt.plot(np.arange(surro.y_min, surro.y_max+(surro.y_max-surro.y_min)/10,(surro.y_max-surro.y_min)/10), np.arange(surro.y_min, surro.y_max+(surro.y_max-surro.y_min)/10,(surro.y_max-surro.y_min)/10), c='black', lw=2)
        p_train = plt.scatter(y_train, y_train_pred, s=50, marker='o', c='none', edgecolors='black')
        p_test = plt.scatter(y_test, y_test_pred, s=50, marker='x', c='black')
        plt.tick_params(labelsize=25) # axis number size
        plt.legend(handles=[p_train, p_test], labels=['Training Data Set', 'Testing Data Set'], loc='best',
                   edgecolor='black', prop=font)
        # plt.show()
        plt.savefig('plot/pred_true_compare_'+str(i)+'.eps')
        plt.close()



class random(cross_validation):
    def divide_surro_0(self):
        surro = self.surro_list[0]  # the first surrogate model
        rand_seq = np.random.permutation(np.arange(0, surro.normalized_X.shape[0], 1))  # generate a random sequence, e.g. if total number of samples: n=5, then  rand_seq=[1,3,2,5,4]
        self.divide_according_to_seq(rand_seq)

    def divide_according_to_seq(self,seq):
        surro = self.surro_list[0] # the first surrogate model
        part_X = np.zeros((self.samples_per_fold, surro.normalized_X.shape[1])) # initialize the first part
        part_y = np.zeros((self.samples_per_fold, surro.normalized_y.shape[1]))

        surro.normalized_fold_X = []  # store all the part_X results
        surro.normalized_fold_y = []
        ind = 0      # index of the sample in part_X
        count = 0    # count of stored samples
        last_part_flag = 0     # last part flag
        for i in seq:
            part_X[ind] = surro.normalized_X[i]  # the samples in part_X follows the seq in the original dataset: surro.normalized_X
            part_y[ind] = surro.normalized_y[i]
            ind += 1
            count += 1

            if (count==self.samples_per_fold*(self.num_fold-1)): # the second-to-last part_X is full, so the initialization of part_X is different
                surro.normalized_fold_X.append(part_X)   # store this part_X
                surro.normalized_fold_y.append(part_y)
                num_rest_samples=surro.normalized_X.shape[0]-self.samples_per_fold*(self.num_fold-1) #numnber of rest samples
                part_X = np.zeros((num_rest_samples, surro.normalized_X.shape[1]))  # initialize according to the numnber of rest samples
                part_y = np.zeros((num_rest_samples, surro.normalized_y.shape[1]))
                ind=0  # initialize index of part_X, and will not run the following code:<if (ind==self.samples_per_fold)>
                last_part_flag=1 # last part flag

            if (count == surro.normalized_X.shape[0]): # all the samples has been stored
                surro.normalized_fold_X.append(part_X) # store this part_X
                surro.normalized_fold_y.append(part_y)
                ind = 0  # initialize index of part_X,, and will not run the following code:<if (ind==self.samples_per_fold)>

            if (ind==self.samples_per_fold) and not last_part_flag: # this part_X is full and is not the last part
                surro.normalized_fold_X.append(part_X)   # store this part_X
                surro.normalized_fold_y.append(part_y)
                part_X = np.zeros((self.samples_per_fold, surro.normalized_X.shape[1]))  # must initialize again, or the values in normalized_fold_X will also change if part_X change
                part_y = np.zeros((self.samples_per_fold, surro.normalized_y.shape[1]))
                ind=0  # initialize index of part_X



class opt_test_data(cross_validation):
    def divide_surro_0(self):
        surro = self.surro_list[0]  # the first surrogate model
        seq = [] # initialize sequence for self.divide_according_to_seq

        normalized_X_tmp = surro.normalized_X.copy()
        for k in range(self.num_fold-1):
            rand_ind = np.random.randint(0, normalized_X_tmp.shape[0]) # generate a random index
            part_X = normalized_X_tmp[rand_ind] # add this x to part_X
            normalized_X_tmp = np.delete(normalized_X_tmp, rand_ind, axis=0) # delete this x in normalized_X_tmp

            for j in range(self.samples_per_fold-1):
                dis=0
                for i in range(normalized_X_tmp.shape[0]):
                    part_X_tmp = np.vstack((part_X,normalized_X_tmp[i])) # stack next x
                    dis_temp = 1 / self.calculate_distance_phi(part_X_tmp) # calculate distance, the larger the better
                    if dis_temp > dis:
                        dis = dis_temp
                        best_ind = i # save best index
                part_X = np.vstack((part_X,normalized_X_tmp[best_ind]))
                normalized_X_tmp = np.delete(normalized_X_tmp, best_ind, axis=0)  # delete this x in normalized_X_tmp

            # part_X full, save seq
            for p in range(part_X.shape[0]):
                for q in range(surro.normalized_X.shape[0]):
                    if (part_X[p, :] == surro.normalized_X[q, :]).all():
                        seq.append(q)
                        break

        # save seq for the last part_X
        for p in range(normalized_X_tmp.shape[0]):
            for q in range(surro.normalized_X.shape[0]):
                if (normalized_X_tmp[p, :] == surro.normalized_X[q, :]).all():
                    seq.append(q)
                    break

        seq=np.array(seq)
        self.divide_according_to_seq(seq)

    # note: the following code is the same in class random(), which can be improved further
    def divide_according_to_seq(self,seq):
        surro = self.surro_list[0]  # the first surrogate model
        part_X = np.zeros((self.samples_per_fold, surro.normalized_X.shape[1])) # initialize the first part
        part_y = np.zeros((self.samples_per_fold, surro.normalized_y.shape[1]))

        surro.normalized_fold_X = []  # store all the part_X results
        surro.normalized_fold_y = []
        ind = 0      # index of the sample in part_X
        count = 0    # count of stored samples
        last_part_flag = 0     # last part flag
        for i in seq:
            part_X[ind] = surro.normalized_X[i]  # the samples in part_X follows the seq in the original dataset: surro.normalized_X
            part_y[ind] = surro.normalized_y[i]
            ind += 1
            count += 1

            if (count==self.samples_per_fold*(self.num_fold-1)): # the second-to-last part_X is full, so the initialization of part_X is different
                surro.normalized_fold_X.append(part_X)   # store this part_X
                surro.normalized_fold_y.append(part_y)
                num_rest_samples=surro.normalized_X.shape[0]-self.samples_per_fold*(self.num_fold-1) #numnber of rest samples
                part_X = np.zeros((num_rest_samples, surro.normalized_X.shape[1]))  # initialize according to the numnber of rest samples
                part_y = np.zeros((num_rest_samples, surro.normalized_y.shape[1]))
                ind=0  # initialize index of part_X, and will not run the following code:<if (ind==self.samples_per_fold)>
                last_part_flag=1 # last part flag

            if (count == surro.normalized_X.shape[0]): # all the samples has been stored
                surro.normalized_fold_X.append(part_X) # store this part_X
                surro.normalized_fold_y.append(part_y)
                ind = 0  # initialize index of part_X,, and will not run the following code:<if (ind==self.samples_per_fold)>

            if (ind==self.samples_per_fold) and not last_part_flag: # this part_X is full and is not the last part
                surro.normalized_fold_X.append(part_X)   # store this part_X
                surro.normalized_fold_y.append(part_y)
                part_X = np.zeros((self.samples_per_fold, surro.normalized_X.shape[1]))  # must initialize again, or the values in normalized_fold_X will also change if part_X change
                part_y = np.zeros((self.samples_per_fold, surro.normalized_y.shape[1]))
                ind=0  # initialize index of part_X


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



# e.g.
if __name__ == '__main__':
    # ==================================================
    rootpath = r'C:\Users\tomjj\Desktop\demo'  # your saopy file path
    import sys

    sys.path.append(rootpath)  # you can directly import the modules in this folder
    sys.path.append(rootpath + r'\saopy\surrogate_model')
    sys.path.append(rootpath + r'\saopy')
    # ==================================================
    from saopy.surrogate_model.ANN import *
    from saopy.surrogate_model.KRG import *
    from saopy.surrogate_model.RBF import *
    from saopy.surrogate_model.surrogate_model import *


    lower_bound = [-32.768, -32.768]
    upper_bound = [32.768, 32.768]
    parallel_num = 3
    num_fold = 3
    surro_list = []
    surro_list.append(ANN(num_layers=2, num_neurons=50))
    # surro_list.append(RBF(num_centers=10))
    # surro_list.append(KRG()) # may take longer training time
    for surro in surro_list:
        surro.load_data(lower_bound, upper_bound)
        surro.normalize_all()
    cros_valid = cross_validation.opt_test_data(surro_list, num_fold=num_fold)
    cros_valid.divide()
    best_ind = cros_valid.begin_cross_validation(parallel_num, plot_flag=1)  # get best model index
    # best_surro = surro_list[best_ind]  # get best model
    # best_surro.train(best_surro.normalized_X, best_surro.normalized_y)  # train again using all samples
    # save_obj(best_surro, 'best_surro')  # save best model