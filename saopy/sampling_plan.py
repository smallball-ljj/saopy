# ==================================================
# original author: from pyKriging.samplingplan
# https://github.com/capaulson/pyKriging
# ==================================================
# improved by luojiajie
# ==================================================

"""
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
Note: samples are generated on [0,1]^d with default, use inverse_norm() to inverse normalize samples according to given range
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

current available sampling plans:
1.full factorial
2.random Latin hypercube sampling
3.optimized Latin hypercube sampling
"""

import os
import numpy as np
import math as m
import matplotlib.pyplot as plt


class sampling_plan():
    """
    base class for all sampling plans
    """
    def __init__(self,number,dimension):
        """
        :param number: number of samples
        (for Latin hypercube sampling, this means total number of samples)
        (for full factorial sampling, this means number of points for each dimension)
        :param dimension: number of design variables (dimensions)
        """
        self.n = number
        self.d = dimension
        self.X = None # sampling results (generated after calling 'begin_sampling()')


    def begin_sampling(self):
        """
        detailed sampling plans, which is defined in the following classes
        """
        pass


    def inverse_norm(self,lower_bound,upper_bound):
        """
        inverse normalize self.X according to lower_bound and upper_bound (e.g.lower_bound=[5,10] upper_bound=[10,100])
        """
        for i in range(self.X.shape[1]):
            self.X[:,i]=self.X[:,i]*(upper_bound[i]-lower_bound[i])+lower_bound[i]
        return self.X


    def output(self,file_name='X.csv'):
        """
        output self.X to <file_name>
        """
        np.savetxt(file_name, self.X, delimiter=',')


    def plot(self, lower_bound, upper_bound):
        """
         currently only available for plotting scatter sampling points
         and the minimum distance between each point and all points
        """
        if os.path.exists('plot') == False:
            os.makedirs('plot')
        dimension=len(lower_bound)
        # ==================================================
        # plot minimum distance between each point and all points
        # ==================================================
        min_dis = np.zeros((self.X.shape[0],1)) # initialize minimum distance
        X_copy=self.X.copy()
        for i in range(self.X.shape[0]):
            dis=np.sum((X_copy-self.X[i])**2,axis=1)**0.5 # calculate the distance^2 between the i th points and all the X. note: matrix calculation is much faster than using 'for'
            dis[dis.argmin()] = dis.max() # note the distance between the point and itself is also calculated, so the minimum distance is zero. we assign this to the maximum value so that we can get the minimum distance
            min_dis[i] = dis.min() # select the minimum distance

        plt.figure(figsize=(10, 8))
        plt.plot(range(1,self.X.shape[0]+1),min_dis)
        plt.xlabel('sample index')
        plt.ylabel('min distance')
        # plt.ylim(0, 0.3)

        # plt.show()
        plt.savefig('plot/initial_sampling_plan_min_distance.eps')
        plt.close()

        # ==================================================
        # 1D scatter
        # ==================================================
        if dimension == 1:
            font = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }
            plt.figure(figsize=(10, 8))
            plt.xlabel('X', font)
            plt.scatter(self.X[:, 0], np.zeros((self.X.shape[0],1)), s=5, marker='o', c='black') # plot scatter of all samples X
            plt.xlim(lower_bound[0], upper_bound[0])  # x range
            # coordinate font
            plt.tick_params(labelsize=25)


        # ==================================================
        # multi dimension scatter
        # ==================================================
        elif dimension >= 2:
            fig = plt.figure(figsize=(10, 7.5))  # figure size
            for i in range(dimension):
                for j in range(dimension - 1, i, -1):
                    col_ind = i  # column index
                    row_ind = -j + dimension - 1  # row index
                    ind = row_ind * (dimension - 1) + col_ind + 1  # subplot index
                    plt.subplot(dimension - 1, dimension - 1, ind)  # locate subplot
                    self.plot_md(i, j)
                    # plot axis label
                    fontXY = {'family': 'Times New Roman', 'weight': 'normal', 'size': 30, }  # xy label size
                    axisfont = 20  # axis number size
                    if row_ind == 0 and col_ind == 0:  # first subplot, plot both axis
                        ax = plt.gca()
                        ax.xaxis.set_ticks_position('top')
                        ax.xaxis.set_label_position('top')
                        plt.xlabel('x' + str(i), fontXY)
                        plt.ylabel('x' + str(j), fontXY)
                    elif row_ind == 0 and col_ind != 0:  # first row, plot x axis label
                        plt.yticks([])
                        ax = plt.gca()
                        ax.xaxis.set_ticks_position('top')
                        ax.xaxis.set_label_position('top')
                        plt.xlabel('x' + str(i), fontXY)
                    elif col_ind == 0 and row_ind != 0:  # first column, plot y axis label
                        plt.xticks([])
                        plt.ylabel('x' + str(j), fontXY)
                    else:  # other subplot, do not show x,y axis label
                        plt.xticks([])
                        plt.yticks([])
                    plt.xlim(lower_bound[i], upper_bound[i])  # x range
                    plt.ylim(lower_bound[j], upper_bound[j])  # x range
                    plt.tick_params(labelsize=axisfont)  # axis number size

        # plt.show()
        plt.savefig('plot/initial_sampling_plan_scatter.eps')
        plt.close()


    def plot_md(self, x1_arg, x2_arg):
        """
        plot multi dimension scatter of the selected two variables: x1_arg and x2_arg

        e.g. x1_arg=0, x2_arg=3, it will plot the scatter with the first and the fourth parameters as variables
        """
        plt.scatter(self.X[:, x1_arg], self.X[:, x2_arg], s=5, marker='o', c='black')  # scatter of all samples X



# ====================================================================================================
# detailed sampling plans
# ====================================================================================================
class fullfactorial(sampling_plan):
    def begin_sampling(self):
        ix = (slice(0, 1, self.n*1j),) * self.d
        self.X = np.mgrid[ix].reshape(self.d, self.n**self.d).T
        return self.X


class random_lhs(sampling_plan):
    def begin_sampling(self,Edges=0):
        """
        Generates a random latin hypercube sampling within the [0,1]^d hypercube

        Inputs:
            Edges-if Edges=1 the extreme bins will have their centers on the edges of the domain
        Outputs:
            Latin hypercube sampling plan of n samples in d dimensions
        """
        #pre-allocate memory
        X = np.zeros((self.n,self.d))

        for i in range(0,self.d):
            X[:,i] = np.transpose(np.random.permutation(np.arange(1,self.n+1,1)))   # e.g. if total number of samples: n=5, then  X[:,i]=[1,3,2,5,4]

        if Edges == 1:
            X = (X-1)/(self.n-1)
        else:
            X = (X-0.5)/self.n

        self.X=X
        return self.X


class optimal_lhs(sampling_plan):
    def begin_sampling(self, p = 2, population=30, iterations=30, q = [2]):
            """
            Generates an optimized Latin hypercube sampling by optimizing the Morris-Mitchell
            criterion for a range of exponents and plots the first two dimensions of
            the current hypercube throughout the optimization process.

            Inputs:
                p - the distance metric to be used (p=1 rectangular (fast) (e.g. for 2D: distance=|x1-x2|+|y1-y2|), p=2 Euclidean)
                Population - number of individuals in the evolutionary operation optimizer
                Iterations - number of generations the evolutionary operation optimizer is run for
                Note: high values for the two inputs above will ensure high quality hypercubes, but the search will take longer.
                q - list of q to optimise Phi_q for (see mmphi())
            Output:
                X - optimized Latin hypercube
            """
            #list of q to optimise Phi_q for (see mmphi())
            # q = [1,2,5,10,20,50,100]

            #we start with a random Latin hypercube sampling
            XStart = random_lhs(self.n,self.d).begin_sampling()

            X3D = np.zeros((self.n,self.d,len(q)))
            #for each q optimize Phi_q
            for i in range(len(q)):
                # print(('Now_optimizing_for_q = %d \n' %q[i]))
                X3D[:,:,i] = self.mmlhs(XStart, population, iterations, q[i],p)

            #sort according to the Morris-Mitchell criterion
            Index = self.mmsort(X3D,p)
            # print(('Best_lh_found_using_q = %d \n' %q[Index[0]]))

            #and the Latin hypercube with the best space-filling properties is
            self.X = X3D[:,:,Index[0]]

            return self.X

    def mmlhs(self, X_start, population,iterations, q,p):
        """
        Evolutionary operation search for the most space filling Latin hypercube
        """
        X_s = X_start.copy()

        n = np.size(X_s,0)

        X_best = X_s

        Phi_best = self.mmphi(X_best,q,p)

        leveloff = m.floor(0.85*iterations)

        for it in range(0,iterations):
            if it < leveloff:
                mutations = int(round(1+(0.5*n-1)*(leveloff-it)/(leveloff-1)))
            else:
                mutations = 1

            X_improved  = X_best
            Phi_improved = Phi_best

            for offspring in range(0,population):
                X_try = self.perturb(X_best, mutations)
                Phi_try = self.mmphi(X_try, q,p)

                if Phi_try < Phi_improved:
                    X_improved = X_try
                    Phi_improved = Phi_try

            if Phi_improved < Phi_best:
                X_best = X_improved
                Phi_best = Phi_improved

        return X_best

    def mmphi(self,X,q,p):
        """
        Calculates the sampling plan quality criterion of Morris and Mitchell

        Inputs:
            X - Sampling plan
            q - exponent used in the calculation of the metric (default = 2)
            p - the distance metric to be used (p=1 rectangular - default , p=2 Euclidean)

        Output:
            Phiq - sampling plan 'space-fillingness' metric
        """
        #calculate the distances between all pairs of points (using the p-norm) and build multiplicity array J
        J,d = self.jd(X,p)
        #the sampling plan quality criterion
        Phiq = (np.sum(J*(d**(-q))))**(1.0/q)
        return Phiq

    def jd(self, X,p):
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

    def perturb(self,X,PertNum):
        """
        Interchanges pairs of randomly chosen elements within randomly
        chosen columns of a sampling plan a number of times. If the plan is
        a Latin hypercube, the result of this operation will also be a Latin
        hypercube.

        Inputs:
            X - sampling plan
            PertNum - the number of changes (perturbations) to be made to X.
        Output:
            X - perturbed sampling plan
        """
        X_pert = X.copy()
        [n,k] = np.shape(X_pert)

        for pert_count in range(0,PertNum):
            col = int(m.floor(np.random.rand(1)*k))  #choose a random column index

            #Choosing two distinct random points
            el1 = 0
            el2 = 0
            while el1 == el2:
                el1 = int(m.floor(np.random.rand(1)*n))
                el2 = int(m.floor(np.random.rand(1)*n))

            #swap the two chosen elements
            arrbuffer = X_pert[el1,col]
            X_pert[el1,col] = X_pert[el2,col]
            X_pert[el2,col] = arrbuffer

        return X_pert

    def mmsort(self,X3D,p):
        """
        Ranks sampling plans according to the Morris-Mitchell criterion definition.
        Note: similar to phisort, which uses the numerical quality criterion Phiq
        as a basis for the ranking.

        Inputs:
            X3D - three-dimensional array containing the sampling plans to be ranked.
            p - the distance metric to be used (p=1 rectangular - default, p=2 Euclidean)
        Output:
            Index - index array containing the ranking
        """
        #Pre-allocate memory
        Index = np.arange(np.size(X3D,axis=2))

        #Bubble-sort
        i = 0
        while i<=len(Index)-2:
            if self.mm(X3D[:,:,Index[i]],X3D[:,:,Index[i+1]],p) == 2:
                arrbuffer=Index[i]
                Index[i] = Index[i+1]
                Index[i+1] = arrbuffer
            i = i + 1
        return Index

    def mm(self,X1,X2,p):
        """
        Given two sampling plans chooses the one with the better space-filling properties
        (as per the Morris-Mitchell criterion)

        Inputs:
            X1,X2 - the two sampling plans
            p - the distance metric to be used (p=1 rectangular-default, p=2 Euclidean)
        Outputs:
            Mmplan - if Mmplan=0, identical plans or equally space-filling,
            if Mmplan=1, X1 is more space filling,
            if Mmplan=2, X2 is more space filling
        """

        #thats how two arrays are compared in their sorted form
        v = np.sort(X1) == np.sort(X2)
        if 	v.all() == True:#if True, then the designs are the same
            return 0
        else:
            #calculate the distance and multiplicity arrays
            [J1 , d1] = self.jd(X1,p);m1=len(d1)
            [J2 , d2] = self.jd(X2,p);m2=len(d2)

            #blend the distance and multiplicity arrays together for
            #comparison according to definition 1.2B. Note the different
            #signs - we are maximising the d's and minimising the J's.
            V1 = np.zeros((2*m1))
            V1[0:len(V1):2] = d1
            V1[1:len(V1):2] = -J1

            V2 = np.zeros((2*m2))
            V2[0:len(V2):2] = d2
            V2[1:len(V2):2] = -J2

            #the longer vector can be trimmed down to the length of the shorter one
            m = min(m1,m2)
            V1 = V1[0:m]
            V2 = V2[0:m]

            #generate vector c such that c(i)=1 if V1(i)>V2(i), c(i)=2 if V1(i)<V2(i)
            #c(i)=0 otherwise
            c = np.zeros(m)
            for i in range(m):
                if np.greater(V1[i],V2[i]) == True:
                    c[i] = 1
                elif np.less(V1[i],V2[i]) == True:
                    c[i] = 2
                elif np.equal(V1[i],V2[i]) == True:
                    c[i] = 0

            #If the plans are not identical but have the same space-filling
            #properties
            if sum(c) == 0:
                return 0
            else:
                #the more space-filling design (mmplan)
                #is the first non-zero element of c
                i = 0
                while c[i] == 0:
                    i = i+1
                return c[i]


# e.g.
if __name__ == '__main__':
    lower_bound = [-32.768, -32.768]
    upper_bound = [32.768, 32.768]

    # a=fullfactorial(6,2)
    # a.begin_sampling()

    # a = random_lhs(40,2)
    # a.begin_sampling()

    a=optimal_lhs(40,2)
    a.begin_sampling(2,30,100,[2])

    a.inverse_norm(lower_bound, upper_bound)

    a.output('output_sampling_plan.csv')

    a.plot()