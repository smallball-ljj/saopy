# ==================================================
# author:luojiajie
# ==================================================

import csv
import numpy as np


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


def output(X,file_name):
    """
    output X to <file_name>

    :param X: numpy array
    """
    np.savetxt(file_name, X, delimiter=',')


def stack(original_csv, new_csv):
    """
    stack the data in new_csv to original_csv, and output original_csv
    """
    new_data = read_csv_to_np(new_csv)
    try: # if there exist original_csv, stack
        original_data=read_csv_to_np(original_csv)
        new_data=np.vstack((original_data,new_data)) # stack new_data to the end of the original_data
    except:
        pass

    if original_csv=='X': # there should not exist duplicate samples in X
        if new_data.shape[0] != np.unique(new_data, axis=0).shape[0]:
            raise RuntimeError('there exist duplicate samples')

    output(new_data, original_csv)


# e.g.
if __name__=="__main__":
    stack('X.csv', 'X_new.csv')