# ==================================================
# original author: JeremyLinux
# https://github.com/JeremyLinux/PyTorch-Radial-Basis-Function-Layer
# for more information about pytorch, visit https://pytorch.org/
# ==================================================
# modified by luojiajie
# !!!note: RBF_pytorch, which use back propagation, is not the interpolating function
# ==================================================


import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from surrogate_model import surrogate_model


# RBFs
def gaussian(alpha):
    phi = torch.exp(-1*alpha.pow(2))
    return phi

def linear(alpha):
    phi = alpha
    return phi

def quadratic(alpha):
    phi = alpha.pow(2)
    return phi

def inverse_quadratic(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2))
    return phi

def multiquadric(alpha):
    phi = (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def inverse_multiquadric(alpha):
    phi = torch.ones_like(alpha) / (torch.ones_like(alpha) + alpha.pow(2)).pow(0.5)
    return phi

def spline(alpha):
    phi = (alpha.pow(2) * torch.log(alpha + torch.ones_like(alpha)))
    return phi

def poisson_one(alpha):
    phi = (alpha - torch.ones_like(alpha)) * torch.exp(-alpha)
    return phi

def poisson_two(alpha):
    phi = ((alpha - 2*torch.ones_like(alpha)) / 2*torch.ones_like(alpha)) \
    * alpha * torch.exp(-alpha)
    return phi

def matern32(alpha):
    phi = (torch.ones_like(alpha) + 3**0.5*alpha)*torch.exp(-3**0.5*alpha)
    return phi

def matern52(alpha):
    phi = (torch.ones_like(alpha) + 5**0.5*alpha + (5/3) \
    * alpha.pow(2))*torch.exp(-5**0.5*alpha)
    return phi

def basis_func_dict():
    """
    A helper function that returns a dictionary containing each RBF
    """
    bases = {'gaussian': gaussian,
             'linear': linear,
             'quadratic': quadratic,
             'inverse quadratic': inverse_quadratic,
             'multiquadric': multiquadric,
             'inverse multiquadric': inverse_multiquadric,
             'spline': spline,
             'poisson one': poisson_one,
             'poisson two': poisson_two,
             'matern32': matern32,
             'matern52': matern52}
    return bases


# RBF Network
class RBFN(nn.Module):
    """
    Transforms incoming data using a given radial basis function:
    u_{i} = rbf(||x - c_{i}|| / s_{i})

    Arguments:
        input_shape: input shape
        num_centers: number of RBF layer centers

    Shape:
        - Input: (N, input_shape) where N is an arbitrary batch size
        - RBF layer Output: (N, num_centers) where N is an arbitrary batch size
        - Linear layer Output: (N, 1)

    Attributes:
        centres: the learnable centres of shape (num_centers, input_shape).
            The values are initialised from a standard normal distribution.
            Normalising inputs to have mean 0 and standard deviation 1 is
            recommended.
        
        sigmas: the learnable scaling factors of shape (num_centers).
            The values are initialised as ones.
        
        basis_func: the radial basis function used to transform the scaled
            distances.
    """
    def __init__(self, input_shape, num_centers, basis_func=gaussian):
        super().__init__()
        self.input_shape = input_shape
        self.num_centers = num_centers
        self.centres = nn.Parameter(torch.Tensor(num_centers, input_shape))
        self.sigmas = nn.Parameter(torch.Tensor(num_centers))
        self.basis_func = basis_func
        self.reset_parameters()
        self.linear_layers = nn.Linear(num_centers, 1)

    def reset_parameters(self):
        nn.init.normal_(self.centres, 0, 1)
        nn.init.constant_(self.sigmas, 1)

    def forward(self, input):
        size = (input.size(0), self.num_centers, self.input_shape)
        x = input.unsqueeze(1).expand(size)
        c = self.centres.unsqueeze(0).expand(size)
        distances = (x - c).pow(2).sum(-1).pow(0.5) * self.sigmas.unsqueeze(0)
        rbf_layer_out = self.basis_func(distances) # shape(N, num_centers)
        linear_layer_out = self.linear_layers(rbf_layer_out) # shape(N, 1)
        return linear_layer_out



class RBF_pytorch(surrogate_model):
    def __init__(self, num_centers=10):
        super().__init__()
        self.num_centers = num_centers


    def train(self, X_train, y_train, plot_loss=0):
        self.rbfn=RBFN(input_shape=X_train.shape[1],num_centers=self.num_centers) # instantiate RBFN
        loss_criterion=nn.MSELoss() # use mean squared error loss
        optimizer = torch.optim.Adadelta(self.rbfn.parameters())

        epoch=10000
        loss_history=[]
        for iter in range(0, epoch):
            optimizer.zero_grad() # zero the gradient, or it will be accumulated
            self.y_pred = self.rbfn(torch.from_numpy(X_train).float()) # forward propagation
            loss = loss_criterion(self.y_pred, torch.from_numpy(y_train).float()) # calculate loss
            loss.backward() # backward propagation
            optimizer.step() # update weight

            loss_history.append(loss.detach().numpy()) # record loss

            # if (iter % 100 == 0):
            #     print(loss)

        if plot_loss==1:
            plt.plot(range(1,epoch+1),loss_history)
            plt.xlabel('epoch')
            plt.ylabel('MSE of training data')
            plt.savefig('plot/RBF_train_history.eps')
            plt.close


    def calculate(self, X):
        """
        :param X: numpy array, with shape(number,dimension)
        """
        X=self.normalize_X(X)
        X=torch.from_numpy(X).float() # convert numpy to tensor
        y=self.rbfn(X)
        y=y.detach().numpy() # convert tensor to numpy
        y=self.inverse_normalize_y(y)
        return y