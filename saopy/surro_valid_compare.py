import csv
import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from matplotlib.pyplot import MultipleLocator


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


def surro_valid_compare(obj_num,total_iter,maxormin,plot_flag):
    """
    compare optimization results using surrogate model and validation using optimized X
    :param obj_num: objective number
    :param total_iter: total outer iteration
    """
    if os.path.exists('plot') == False:
        os.makedirs('plot')

    if obj_num == 1: # compare results of single objective optimization
        best_var = read_csv_to_np('best_var.csv')
        best_var=best_var[0:total_iter,:]
        X = read_csv_to_np('X.csv')
        y = read_csv_to_np('y0.csv')

        best_ObjV_valid = np.zeros((best_var.shape[0], 1))
        div_ind = []  # record CFD divergence case
        for i in range(best_var.shape[0]):
            for j in range(X.shape[0]):
                if (X[j, :] == best_var[i, :]).all():
                    best_ObjV_valid[i, :] = y[j, :]
                    break
                elif j==X.shape[0]-1: # cannot find the same X optimized_X, because CFD diverge
                        div_ind.append(i) # record CFD divergence case
        np.savetxt('best_ObjV_valid.csv', best_ObjV_valid, delimiter=',')

        best_ObjV_surro = read_csv_to_np('best_ObjV.csv')
        best_ObjV_surro = best_ObjV_surro[0:total_iter, :]

        ###########################
        best_ObjV_valid_index=np.arange(1, best_ObjV_valid.shape[0] + 1) # validation index
        for i in range(len(div_ind)):  # remove CFD divergence case
            rm_ind = div_ind[i] - i  # the index to remove, note: when one index is removed, the total index in the array will -1 after that index
            best_ObjV_valid = np.delete(best_ObjV_valid, rm_ind, 0)
            best_ObjV_valid_index = np.delete(best_ObjV_valid_index, rm_ind, 0)
        ###########################

        if plot_flag==1:
            plt.figure(figsize=(10, 8))
            plt.plot(best_ObjV_valid_index, best_ObjV_valid, marker='^', c='black')
            plt.plot(range(1, best_ObjV_surro.shape[0] + 1), best_ObjV_surro, '--', marker='s', c='black')
            plt.tick_params(labelsize=25)  # axis number size
            fontXY = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
            font_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
            plt.xlabel('iteration', fontXY)
            plt.ylabel('min y', fontXY)

            x_major_locator = MultipleLocator(1)# 把x轴的刻度间隔设置为1，并存在变量里
            ax = plt.gca()
            ax.xaxis.set_major_locator(x_major_locator)

            plt.legend(labels=['Actual objective values from real functions', 'optimized y using surrogate model'], loc='best', edgecolor='black', prop=font_legend)
            # plt.show()
            plt.savefig('plot/surro_valid_compare.eps')
            plt.close()


        # #best so far(max version) ##########################
        # if maxormin == -1:
        #     best_so_far = []
        #     best_so_far.append(best_ObjV_valid[0])  # append the first ObjV_valid
        #     for i in range(1, best_ObjV_surro.shape[0]):
        #         for j in range(1, best_ObjV_valid_index.shape[0]):
        #             if best_ObjV_valid_index[j] == i + 1:
        #                 if best_so_far[i - 1] < best_ObjV_valid[j]:
        #                     best_so_far.append(best_ObjV_valid[j])
        #                 else:
        #                     best_so_far.append(best_so_far[i - 1])
        #                 break
        #
        #         if len(best_so_far) < i + 1:
        #             best_so_far.append(best_so_far[i - 1])

        #best so far(min version) ##########################
        if maxormin==1:
            best_so_far=[]
            best_so_far.append(best_ObjV_valid[0]) #append the first ObjV_valid
            for i in range(1,best_ObjV_surro.shape[0]):
                for j in range(1,best_ObjV_valid_index.shape[0]):
                    if best_ObjV_valid_index[j]==i+1:
                        if best_so_far[i-1]>best_ObjV_valid[j]:
                            best_so_far.append(best_ObjV_valid[j])
                        else:
                            best_so_far.append(best_so_far[i-1])
                        break

                if len(best_so_far)<i+1:
                    best_so_far.append(best_so_far[i - 1])

        if plot_flag==1:
            plt.plot(range(1, len(best_so_far)+1), best_so_far, '-', marker='o', c='black')
            plt.tick_params(labelsize=25)  # axis number size
            fontXY = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
            font_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 15, }
            plt.xlabel('iteration', fontXY)
            plt.ylabel('best so far', fontXY)

            x_major_locator = MultipleLocator(5)  # 把x轴的刻度间隔设置为1，并存在变量里
            ax = plt.gca()
            ax.xaxis.set_major_locator(x_major_locator)

            plt.legend(labels=['Best actual objective values\nfrom CFD simulations'],
                       loc='best', edgecolor='black', prop=font_legend)
            plt.subplots_adjust(top=0.88, bottom=0.2, left=0.25, right=0.98)
            # plt.show()
            plt.savefig('plot/best_so_far.eps')
            plt.close()

        np.savetxt('plot/best_so_far.csv', np.array(best_so_far), delimiter=',')






    elif obj_num == 2: # compare results of 2 objective optimization
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
            p1=plt.scatter(y0_opt_valid, y1_opt_valid, s=50, marker='^', c='none', edgecolors='black')

            y_opt_surro = np.loadtxt('Result_' + str(iter) + '/ObjV.csv', delimiter=',')
            y0_opt_surro = y_opt_surro[:, 0]
            y1_opt_surro = y_opt_surro[:, 1]
            p2=plt.scatter(y0_opt_surro, y1_opt_surro, s=50, marker='s', c='none', edgecolors='black')

            for i in range(y0_opt_valid.shape[0]): # plot line between opt_surro and opt_surro
                plt.plot(np.array([y0_opt_valid[i],y0_opt_surro[i]]),np.array([y1_opt_valid[i],y1_opt_surro[i]]),'--',c='black')

            fontXY = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
            font_legend = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
            plt.xlabel('y0', fontXY)
            plt.ylabel('y1', fontXY)
            plt.tick_params(labelsize=25)  # axis number size
            plt.legend(handles=[p1,p2],labels=['Actual objective values from real functions', 'Pareto front points using surrogate model'], loc='best', edgecolor='black', prop=font_legend)
            # plt.show()
            plt.savefig('plot/surro_valid_compare_'+str(iter)+'.eps')
            plt.close()


    elif obj_num == 3: # compare results of 3 objective optimization
        total_iter -= 1  # note: the last iteration X_new.csv is not stack to X.csv

        y0_opt_valid_min=[]
        y1_opt_valid_min = []
        y2_opt_valid_min = []

        X = read_csv_to_np('X.csv')
        y0 = read_csv_to_np('y0.csv')
        y1 = read_csv_to_np('y1.csv')
        y2 = read_csv_to_np('y2.csv')

        for iter in range(total_iter):
            optimized_X = read_csv_to_np('Result_' + str(iter) + '/Phen.csv')

            y0_opt_valid = np.zeros((optimized_X.shape[0], 1))
            y1_opt_valid = np.zeros((optimized_X.shape[0], 1))
            y2_opt_valid = np.zeros((optimized_X.shape[0], 1))
            div_ind = []  # record CFD divergence case
            for i in range(optimized_X.shape[0]):
                for j in range(X.shape[0]):
                    if (X[j, :] == optimized_X[i, :]).all():
                        y0_opt_valid[i, :] = y0[j, :]
                        y1_opt_valid[i, :] = y1[j, :]
                        y2_opt_valid[i, :] = y2[j, :]
                        break
                    elif j == X.shape[0] - 1:  # cannot find the same X optimized_X, because CFD diverge
                        div_ind.append(i)  # record CFD divergence case

            y_opt_surro = np.loadtxt('Result_' + str(iter) + '/ObjV.csv', delimiter=',')
            y0_opt_surro = y_opt_surro[:, 0]
            y1_opt_surro = y_opt_surro[:, 1]
            y2_opt_surro = y_opt_surro[:, 2]

            ###########################
            for i in range(len(div_ind)): # remove CFD divergence case
                rm_ind = div_ind[i] - i # the index to remove, note: when one index is removed, the total index in the array will -1 after that index
                y0_opt_valid = np.delete(y0_opt_valid, rm_ind, 0)
                y1_opt_valid = np.delete(y1_opt_valid, rm_ind, 0)
                y2_opt_valid = np.delete(y2_opt_valid, rm_ind, 0)
                y0_opt_surro = np.delete(y0_opt_surro, rm_ind, 0)
                y1_opt_surro = np.delete(y1_opt_surro, rm_ind, 0)
                y2_opt_surro = np.delete(y2_opt_surro, rm_ind, 0)
            ###########################

            fig = plt.figure(figsize=(10, 8))

            grid = plt.GridSpec(5, 2, wspace=0.5, hspace=0.8)

            # 3D pareto front
            # ax = Axes3D(fig)
            # ax=fig.add_subplot(2, 2, 1, projection='3d')
            ax = fig.add_subplot(grid[1:3, 0], projection='3d')

            p1=ax.scatter(y0_opt_valid, y1_opt_valid, y2_opt_valid, marker='^', c='b')
            p2=ax.scatter(y0_opt_surro, y1_opt_surro, y2_opt_surro, marker='s', c='r')

            for i in range(y0_opt_valid.shape[0]): # plot line between opt_surro and opt_surro
                ax.plot(np.array([y0_opt_valid[i],y0_opt_surro[i]]),np.array([y1_opt_valid[i],y1_opt_surro[i]]),np.array([y2_opt_valid[i],y2_opt_surro[i]]),'--',c='black')

            ax.set_zlabel('F3', fontdict={'size': 20, 'color': 'black'})
            ax.set_ylabel('F2', fontdict={'size': 20, 'color': 'black'})
            ax.set_xlabel('F1', fontdict={'size': 20, 'color': 'black'})
            ax.view_init(elev=30, azim=30)  # view angle
            # ax1 = plt.gca() # scientific notation
            # ax1.xaxis.get_major_formatter().set_powerlimits((0, 1))  # scientific notation
            # ax1.yaxis.get_major_formatter().set_powerlimits((0, 1)) # scientific notation
            # ax1.zaxis.get_major_formatter().set_powerlimits((0, 1))  # scientific notation

            # y0 vs y1
            ax = fig.add_subplot(grid[1:3, 1])
            p1 = ax.scatter(y0_opt_valid, y1_opt_valid, marker='^', c='b')
            p2 = ax.scatter(y0_opt_surro, y1_opt_surro, marker='s', c='r')

            for i in range(y0_opt_valid.shape[0]):  # plot line between opt_surro and opt_surro
                ax.plot(np.array([y0_opt_valid[i], y0_opt_surro[i]]), np.array([y1_opt_valid[i], y1_opt_surro[i]]), '--', c='black')

            ax.set_ylabel('F2', fontdict={'size': 20, 'color': 'black'})
            ax.set_xlabel('F1', fontdict={'size': 20, 'color': 'black'})
            plt.tick_params(labelsize=17)
            # ax1 = plt.gca() # scientific notation
            # ax1.xaxis.get_major_formatter().set_powerlimits((0, 1))  # scientific notation
            # ax1.yaxis.get_major_formatter().set_powerlimits((0, 1)) # scientific notation

            # y0 vs y2
            ax = fig.add_subplot(grid[3:5, 0])
            p1 = ax.scatter(y0_opt_valid, y2_opt_valid, marker='^', c='b')
            p2 = ax.scatter(y0_opt_surro, y2_opt_surro, marker='s', c='r')

            for i in range(y0_opt_valid.shape[0]):  # plot line between opt_surro and opt_surro
                ax.plot(np.array([y0_opt_valid[i], y0_opt_surro[i]]), np.array([y2_opt_valid[i], y2_opt_surro[i]]), '--', c='black')

            ax.set_ylabel('F3', fontdict={'size': 20, 'color': 'black'})
            ax.set_xlabel('F1', fontdict={'size': 20, 'color': 'black'})
            plt.tick_params(labelsize=17)
            # ax.set_ylim(min(y2_opt_valid.min(),y2_opt_surro.min()),max(y2_opt_valid.max(),y2_opt_surro.max()))
            # ax1 = plt.gca() # scientific notation
            # ax1.xaxis.get_major_formatter().set_powerlimits((0, 1)) # scientific notation
            # ax1.yaxis.get_major_formatter().set_powerlimits((0, 1)) # scientific notation

            # y1 vs y2
            ax = fig.add_subplot(grid[3:5, 1])
            p1 = ax.scatter(y1_opt_valid, y2_opt_valid, marker='^', c='b')
            p2 = ax.scatter(y1_opt_surro, y2_opt_surro, marker='s', c='r')

            for i in range(y0_opt_valid.shape[0]):  # plot line between opt_surro and opt_surro
                ax.plot(np.array([y1_opt_valid[i], y1_opt_surro[i]]), np.array([y2_opt_valid[i], y2_opt_surro[i]]), '--', c='black')

            ax.set_ylabel('F3', fontdict={'size': 20, 'color': 'black'})
            ax.set_xlabel('F2', fontdict={'size': 20, 'color': 'black'})
            plt.tick_params(labelsize=17)
            # ax.set_ylim(min(y2_opt_valid.min(), y2_opt_surro.min()), max(y2_opt_valid.max(), y2_opt_surro.max()))
            # ax1 = plt.gca() # scientific notation
            # ax1.xaxis.get_major_formatter().set_powerlimits((0, 1))  # scientific notation
            # ax1.yaxis.get_major_formatter().set_powerlimits((0, 1)) # scientific notation

            # legend
            ax = fig.add_subplot(grid[0, 0])
            ax.axis('off')
            font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 20, }
            ax.legend(handles=[p1,p2],labels=['Actual objective values from real functions', 'Pareto front points using surrogate model'], loc='upper left', edgecolor='black', prop=font)

            plt.subplots_adjust(top=1) # remove top blank
            # plt.subplots_adjust(top=1, bottom=0, right=0.93, left=0, hspace=0, wspace=0)
            # plt.show()
            plt.savefig('plot/surro_valid_compare_'+str(iter)+'.eps')
            plt.close()

            y0_opt_valid_min.append(y0_opt_valid.min())
            y1_opt_valid_min.append(y1_opt_valid.min())
            y2_opt_valid_min.append(y2_opt_valid.min())


        # evolution of pareto front - new
        y0_opt_valid_min = np.array(y0_opt_valid_min)
        y1_opt_valid_min = np.array(y1_opt_valid_min)
        y2_opt_valid_min = np.array(y2_opt_valid_min)
        for i in range(1, y0_opt_valid_min.shape[0]):
            if y0_opt_valid_min[i] > y0_opt_valid_min[i - 1]:
                y0_opt_valid_min[i] = y0_opt_valid_min[i - 1]
        for i in range(1, y1_opt_valid_min.shape[0]):
            if y1_opt_valid_min[i] > y1_opt_valid_min[i - 1]:
                y1_opt_valid_min[i] = y1_opt_valid_min[i - 1]
        for i in range(1, y2_opt_valid_min.shape[0]):
            if y2_opt_valid_min[i] > y2_opt_valid_min[i - 1]:
                y2_opt_valid_min[i] = y2_opt_valid_min[i - 1]

        x0 = np.arange(1, total_iter + 1)

        ind = total_iter
        x = x0[0:ind]
        y0 = y0_opt_valid_min[0:ind]
        y1 = y1_opt_valid_min[0:ind]
        y2 = y2_opt_valid_min[0:ind]
        # y0 = (y0 - y0.min()) / (y0.max() - y0.min())  # normalize y
        # y1 = (y1 - y1.min()) / (y1.max() - y1.min())  # normalize y
        # y2 = (y2 - y2.min()) / (y2.max() - y2.min())  # normalize y

        y = [y0, y1, y2]
        F = ['F1', 'F2', 'F3']
        for i in range(len(y)):
            # plt.figure(figsize=(16, 12))
            plt.figure()
            plt.plot(x, y[i], marker='o', c='black')

            x_major_locator = MultipleLocator(10)  # 把x轴的刻度间隔设置为1，并存在变量里
            ax = plt.gca()
            ax.xaxis.set_major_locator(x_major_locator)

            # ax1 = plt.gca()  # scientific notation
            # ax1.yaxis.get_major_formatter().set_powerlimits((0, 1))  # scientific notation

            plt.tick_params(labelsize=30)  # axis number size
            fontXY = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
            plt.xlabel('Iteration', fontXY)
            if i == 0:
                plt.ylabel('Minimum actual ' + F[i] + '\non the Pareto front', fontXY)
            if i == 1:
                plt.ylabel('Minimum actual ' + F[i] + '\non the Pareto front', fontXY)
            if i == 2:
                plt.ylabel('Minimum actual ' + F[i] + '\non the Pareto front', fontXY)

            plt.subplots_adjust(top=0.88, bottom=0.2, left=0.4, right=0.98)
            # plt.show()
            plt.savefig('plot/evolution_of_pareto_front-' + F[i] + '.eps')
            plt.close()