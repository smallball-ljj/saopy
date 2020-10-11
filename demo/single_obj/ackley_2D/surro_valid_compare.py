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
# from saopy.surrogate_model.KRG import *
# from saopy.surrogate_model.RBF import *
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

    surro_valid_compare(obj_num=1, total_iter=20,maxormin=1,plot_flag=1)