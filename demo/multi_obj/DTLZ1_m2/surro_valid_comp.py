# ==================================================
rootpath=r'D:\ljj\aa\demo' # your saopy file path
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
from surro_valid_compare import *

surro_valid_compare(obj_num=3, total_iter=21, maxormin=1, plot_flag=1)