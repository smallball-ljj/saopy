import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


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


def surro_valid_compare(obj_num,total_iter):
    """
    compare optimization results using surrogate model and validation using optimized X
    :param obj_num: objective number
    :param total_iter: total outer iteration
    """
    if os.path.exists('plot') == False:
        os.makedirs('plot')

    if obj_num == 1:
        best_var = read_csv_to_np('best_var.csv')
        X = read_csv_to_np('X.csv')
        y = read_csv_to_np('y.csv')

        best_ObjV_valid = np.zeros((best_var.shape[0], 1))
        for i in range(best_var.shape[0]):
            for j in range(X.shape[0]):
                if (X[j, :] == best_var[i, :]).all():
                    best_ObjV_valid[i, :] = y[j, :]
                    break
        np.savetxt('best_ObjV_valid', best_ObjV_valid, delimiter=',')

        best_ObjV_surro = read_csv_to_np('best_ObjV.csv')

        plt.figure(figsize=(10, 8))
        plt.plot(range(1, best_ObjV_valid.shape[0] + 1), best_ObjV_valid, marker='^', c='black')
        plt.plot(range(1, best_ObjV_surro.shape[0] + 1), best_ObjV_surro, '--', marker='s', c='black')
        plt.tick_params(labelsize=25)  # axis number size
        fontXY = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
        font_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
        plt.xlabel('iteration', fontXY)
        plt.ylabel('min y', fontXY)
        plt.legend(labels=['validation', 'optimized y using surrogate model'], loc='best', edgecolor='black', prop=font_legend)
        # plt.show()
        plt.savefig('plot/surro_valid_compare.eps')
        plt.close()


    elif obj_num == 2:
        total_iter -= 1  # note: the last iteration X_new.csv is not stack to X.csv

        X = read_csv_to_np('X.csv')
        y0 = read_csv_to_np('y0.csv')
        y1 = read_csv_to_np('y1.csv')

        for iter in range(total_iter):
            optimized_X = read_csv_to_np('Result_' + str(iter) + '/Phen.csv')

            y0_opt_valid = np.zeros((optimized_X.shape[0], 1))
            y1_opt_valid = np.zeros((optimized_X.shape[0], 1))
            for i in range(optimized_X.shape[0]):
                for j in range(X.shape[0]):
                    if (X[j, :] == optimized_X[i, :]).all():
                        y0_opt_valid[i, :] = y0[j, :]
                        y1_opt_valid[i, :] = y1[j, :]
                        break

            plt.figure(figsize=(10, 8))
            plt.scatter(y0_opt_valid, y1_opt_valid, s=50, marker='^', c='none', edgecolors='black')

            y_opt_surro = np.loadtxt('Result_' + str(iter) + '/ObjV.csv', delimiter=',')
            y0_opt_surro = y_opt_surro[:, 0]
            y1_opt_surro = y_opt_surro[:, 1]
            plt.scatter(y0_opt_surro, y1_opt_surro, s=50, marker='s', c='none', edgecolors='black')
            fontXY = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
            font_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
            plt.xlabel('y0', fontXY)
            plt.ylabel('y1', fontXY)
            plt.tick_params(labelsize=25)  # axis number size
            plt.legend(labels=['validation', 'pareto front points using surrogate model'], loc='best', edgecolor='black', prop=font_legend)
            # plt.show()
            plt.savefig('plot/surro_valid_compare_'+str(iter)+'.eps')
            plt.close()


    elif obj_num == 3:
        total_iter -= 1  # note: the last iteration X_new.csv is not stack to X.csv

        X = read_csv_to_np('X.csv')
        y0 = read_csv_to_np('y0.csv')
        y1 = read_csv_to_np('y1.csv')
        y2 = read_csv_to_np('y2.csv')

        for iter in range(total_iter):
            optimized_X = read_csv_to_np('Result_' + str(iter) + '/Phen.csv')

            y0_opt_valid = np.zeros((optimized_X.shape[0], 1))
            y1_opt_valid = np.zeros((optimized_X.shape[0], 1))
            y2_opt_valid = np.zeros((optimized_X.shape[0], 1))
            for i in range(optimized_X.shape[0]):
                for j in range(X.shape[0]):
                    if (X[j, :] == optimized_X[i, :]).all():
                        y0_opt_valid[i, :] = y0[j, :]
                        y1_opt_valid[i, :] = y1[j, :]
                        y2_opt_valid[i, :] = y2[j, :]
                        break

            fig = plt.figure(figsize=(10, 8))
            ax = Axes3D(fig)
            ax.scatter(y0_opt_valid, y1_opt_valid, y2_opt_valid, marker='^', c='b')

            y_opt_surro = np.loadtxt('Result_' + str(iter) + '/ObjV.csv', delimiter=',')
            y0_opt_surro = y_opt_surro[:, 0]
            y1_opt_surro = y_opt_surro[:, 1]
            y2_opt_surro = y_opt_surro[:, 2]

            ax.scatter(y0_opt_surro, y1_opt_surro, y2_opt_surro, marker='s', c='r')
            ax.set_zlabel('y0', fontdict={'size': 15, 'color': 'black'})
            ax.set_ylabel('y1', fontdict={'size': 15, 'color': 'black'})
            ax.set_xlabel('y2', fontdict={'size': 15, 'color': 'black'})
            ax.view_init(elev=27, azim=-8)  # view angle

            font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
            ax.legend(labels=['validation', 'pareto front points using surrogate model'], loc='best', edgecolor='black', prop=font)
            # plt.show()
            plt.savefig('plot/surro_valid_compare_'+str(iter)+'.eps')
            plt.close()