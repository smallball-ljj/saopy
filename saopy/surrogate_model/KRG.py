# ==================================================
# original author: from pyKriging.krige & matrixops
# https://github.com/capaulson/pyKriging
# ==================================================
# modified by luojiajie
# ==================================================

import numpy as np
from numpy.matlib import eye
import scipy
from scipy.optimize import minimize
import inspyred
from inspyred import ec
from random import Random
from time import time
import math as m

from surrogate_model import surrogate_model


class matrixops():
    def __init__(self):
        self.LnDetPsi = None
        self.Psi = np.zeros((self.n,self.n), dtype=np.float)
        self.psi = np.zeros((self.n,1))
        self.one = np.ones(self.n)
        self.mu = None
        self.U = None
        self.SigmaSqr = None
        self.Lambda = 1
        self.updateData()

    def updateData(self):
        self.distance = np.zeros((self.n,self.n, self.k))
        for i in range(self.n):
            for j in range(i+1,self.n):
                self.distance[i,j]= np.abs((self.X_train[i]-self.X_train[j]))

    def updatePsi(self):
        self.Psi = np.zeros((self.n,self.n), dtype=np.float)
        self.one = np.ones(self.n)
        self.psi = np.zeros((self.n,1))
        newPsi = np.exp(-np.sum(self.theta*np.power(self.distance,self.pl), axis=2))
        self.Psi = np.triu(newPsi,1)
        self.Psi = self.Psi + self.Psi.T + np.mat(eye(self.n))+np.multiply(np.mat(eye(self.n)),np.spacing(1))
        self.U = np.linalg.cholesky(self.Psi)
        self.U = self.U.T

    def neglikelihood(self):
        self.LnDetPsi=2*np.sum(np.log(np.abs(np.diag(self.U))))

        a = np.linalg.solve(self.U.T, self.one.T)
        b = np.linalg.solve(self.U, a)
        c = self.one.T.dot(b)
        d = np.linalg.solve(self.U.T, self.y_train)
        e = np.linalg.solve(self.U, d)
        self.mu=(self.one.T.dot(e))/c

        self.SigmaSqr = ((self.y_train-self.one.dot(self.mu)).T.dot(np.linalg.solve(self.U,np.linalg.solve(self.U.T,(self.y_train-self.one.dot(self.mu))))))/self.n
        self.NegLnLike=-1.*(-(self.n/2.)*np.log(self.SigmaSqr) - 0.5*self.LnDetPsi)

    def predict_normalized(self,x):
        for i in range(self.n):
            self.psi[i]=np.exp(-np.sum(self.theta*np.power((np.abs(self.X_train[i]-x)),self.pl)))
        z = self.y_train-self.one.dot(self.mu)
        a = np.linalg.solve(self.U.T, z)
        b=np.linalg.solve(self.U, a)
        c=self.psi.T.dot(b)

        f=self.mu + c
        return f[0]



class KRG(matrixops,surrogate_model):
    def __init__(self):
        surrogate_model.__init__(self)


    def train(self, X_train, y_train, optimizer='pso'):
        '''
        The function trains the hyperparameters of the Kriging model.
        :param optimizer: Two optimizers are implemented, a Particle Swarm Optimizer or a GA
        '''
        self.n = X_train.shape[0]
        self.k = X_train.shape[1]
        self.theta = np.ones(self.k)
        self.pl = np.ones(self.k) * 2.
        self.sigma = 0

        self.thetamin = 1e-5
        self.thetamax = 100
        self.pmin = 1
        self.pmax = 2

        self.X_train = X_train
        self.y_train = y_train

        # resize y_train to (1,n), specific usage for kriging defined by original author
        self.y_train.resize((self.y_train.shape[0]))

        matrixops.__init__(self)


        # First make sure our data is up-to-date
        self.updateData()

        # Establish the bounds for optimization for theta and p values
        lowerBound = [self.thetamin] * self.k + [self.pmin] * self.k
        upperBound = [self.thetamax] * self.k + [self.pmax] * self.k

        #Create a random seed for our optimizer to use
        rand = Random()
        rand.seed(int(time()))

        # If the optimizer option is PSO, run the PSO algorithm
        if optimizer == 'pso':
            ea = inspyred.swarm.PSO(Random())
            ea.terminator = self.no_improvement_termination
            ea.topology = inspyred.swarm.topologies.ring_topology
            # ea.observer = inspyred.ec.observers.stats_observer
            final_pop = ea.evolve(generator=self.generate_population,
                                  evaluator=self.fittingObjective,
                                  pop_size=300,
                                  maximize=False,
                                  bounder=ec.Bounder(lowerBound, upperBound),
                                  max_evaluations=30000,
                                  neighborhood_size=20,
                                  num_inputs=self.k)
            # Sort and print the best individual, who will be at index 0.
            final_pop.sort(reverse=True)

        # If not using a PSO search, run the GA
        elif optimizer == 'ga':
            ea = inspyred.ec.GA(Random())
            ea.terminator = self.no_improvement_termination
            final_pop = ea.evolve(generator=self.generate_population,
                                  evaluator=self.fittingObjective,
                                  pop_size=300,
                                  maximize=False,
                                  bounder=ec.Bounder(lowerBound, upperBound),
                                  max_evaluations=30000,
                                  num_elites=10,
                                  mutation_rate=.05)

        # This code updates the model with the hyperparameters found in the global search
        for entry in final_pop:
            newValues = entry.candidate
            locOP_bounds = []
            for i in range(self.k):
                locOP_bounds.append( [self.thetamin, self.thetamax] )

            for i in range(self.k):
                locOP_bounds.append( [self.pmin, self.pmax] )

            # Let's quickly double check that we're at the optimal value by running a quick local optimizaiton
            lopResults = minimize(self.fittingObjective_local, newValues, method='SLSQP', bounds=locOP_bounds, options={'disp': False})

            newValues = lopResults['x']

            # Finally, set our new theta and pl values and update the model again
            for i in range(self.k):
                self.theta[i] = newValues[i]
            for i in range(self.k):
                self.pl[i] = newValues[i + self.k]
            try:
                self.updateModel()
            except:
                pass
            else:
                break


    def updateModel(self):
        '''
        The function rebuilds the Psi matrix to reflect new data or a change in hyperparamters
        '''
        try:
            self.updatePsi()
        except Exception as err:
            #pass
            # print Exception, err
            raise Exception("bad params")

    def generate_population(self, random, args):
        '''
        Generates an initial population for any global optimization that occurs in pyKriging
        :param random: A random seed
        :param args: Args from the optimizer, like population size
        :return chromosome: The new generation for our global optimizer to use
        '''
        size = args.get('num_inputs', None)
        bounder = args["_ec"].bounder
        chromosome = []
        for lo, hi in zip(bounder.lower_bound, bounder.upper_bound):
            chromosome.append(random.uniform(lo, hi))
        return chromosome

    def no_improvement_termination(self, population, num_generations, num_evaluations, args):
        """Return True if the best fitness does not change for a number of generations of if the max number
        of evaluations is exceeded.

        .. Arguments:
           population -- the population of Individuals
           num_generations -- the number of elapsed generations
           num_evaluations -- the number of candidate solution evaluations
           args -- a dictionary of keyword arguments

        Optional keyword arguments in args:

        - *max_generations* -- the number of generations allowed for no change in fitness (default 10)

        """
        max_generations = args.setdefault('max_generations', 10)
        previous_best = args.setdefault('previous_best', None)
        max_evaluations = args.setdefault('max_evaluations', 30000)
        current_best = np.around(max(population).fitness, decimals=4)
        if previous_best is None or previous_best != current_best:
            args['previous_best'] = current_best
            args['generation_count'] = 0
            return False or (num_evaluations >= max_evaluations)
        else:
            if args['generation_count'] >= max_generations:
                return True
            else:
                args['generation_count'] += 1
                return False or (num_evaluations >= max_evaluations)

    def fittingObjective(self,candidates, args):
        '''
        The objective for a series of candidates from the hyperparameter global search.
        :param candidates: An array of candidate design vectors from the global optimizer
        :param args: args from the optimizer
        :return fitness: An array of evaluated NegLNLike values for the candidate population
        '''
        fitness = []
        for entry in candidates:
            f=10000
            for i in range(self.k):
                self.theta[i] = entry[i]
            for i in range(self.k):
                self.pl[i] = entry[i + self.k]
            try:
                self.updateModel()
                self.neglikelihood()
                f = self.NegLnLike
            except Exception as e:
                # print 'Failure in NegLNLike, failing the run'
                # print Exception, e
                f = 10000
            fitness.append(f)
        return fitness

    def fittingObjective_local(self,entry):
        '''
        :param entry: The same objective function as the global optimizer, but formatted for the local optimizer
        :return: The fitness of the surface at the hyperparameters specified in entry
        '''
        f=10000
        for i in range(self.k):
            self.theta[i] = entry[i]
        for i in range(self.k):
            self.pl[i] = entry[i + self.k]
        try:
            self.updateModel()
            self.neglikelihood()
            f = self.NegLnLike
        except Exception as e:
            # print 'Failure in NegLNLike, failing the run'
            # print Exception, e
            f = 10000
        return f


    def calculate(self, X):
        """
        :param X: numpy array, with shape(number,dimension)
        """
        X=self.normalize_X(X)

        pred=[]
        for i in range(X.shape[0]):
            pred.append(self.predict_normalized(X[i]))
        pred=np.array(pred)
        pred.resize((X.shape[0], 1))

        y = self.inverse_normalize_y(pred)
        return y