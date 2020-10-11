# ==================================================
# author:luojiajie
# for more information about RBF_smt, visit
# https://github.com/SMTorg/SMT
# https://smt.readthedocs.io/en/latest/_src_docs/surrogate_models/rbf.html
# ==================================================

from surrogate_model import surrogate_model
from smt.surrogate_models import RBF


class RBF_smt(surrogate_model):
    def __init__(self,d0=1):
        super().__init__()
        self.rbf_sm=RBF(d0=d0)

    def train(self, X_train, y_train):
        self.rbf_sm.set_training_values(X_train, y_train)
        self.rbf_sm.train()

    def calculate(self, X):
        """
        :param X: numpy array, with shape(number,dimension)
        """
        X=self.normalize_X(X)
        y = self.rbf_sm.predict_values(X)
        y = self.inverse_normalize_y(y)
        return y