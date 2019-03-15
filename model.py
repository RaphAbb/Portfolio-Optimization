import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from datetime import datetime

class ADMM_Solver():
    '''
    Implement an ADMM resolution for the convex quadratic portfolio optimization problem
    '''
    def __init__(self, data):
        '''
        Define the return and log_return

        :param data: pd.DataFrame
        '''
        self.data = data
        self.ret = self.data/self.data.shift(1)-1
        self.log_ret = np.log(self.data/self.data.shift(1)).dropna()

    def cov_matrix(self, date, window = 120):
        '''
        Compute the covariance matrix, using pandas implementation

        :param date: pd.Timestamp - date at which we start the time window
        :param window: int - # of days on which we compute the cov matrix
        '''
        data_ = self.data.loc[date:]
        data_ = data_[:window]
        return np.array(data_.cov())

    def cov_matrix_hand(self, date, window = 120):
        '''
        Compute the covariance matrix manually

        :param date: pd.Timestamp - date at which we start the time window
        :param window: int - # of days on which we compute the cov matrix
        '''    	
        data = self.data.loc[:date]
        data = data[date-window:]
        A = np.zeros([10,10])
        sum_ = np.array(data.sum(axis = 0))
        y_bar = sum_/window
        index_ = {0:'SPY',1:'VTI',2:'BND',3:'EMB',4:'TLT',5:'MBB',6:'IAU',7:'USO',8:'USDEUR',9:'USDJPY'}
        for i in range(10):
            for j in range(10):
                if i == j:
                    A[(i,i)] = (np.square(data[index_[i]] - y_bar[i])).sum(axis=0)/window
                else:
                    A[(i,j)] = ((data[index_[i]] - y_bar[i])*(data[index_[j]] - y_bar[j])).sum(axis=0)/window
        return A

    def get_cov(self, by_hand, window, date):
        if by_hand:
            self.cov = self.cov_matrix_hand(window = window, date = date)
            self.sigma = self.cov.diagonal().T
        else:
            self.cov = self.cov_matrix(window = window, date = date)
            self.sigma = self.cov.diagonal().T

    def grad_x(self, x, z, u, rho):
        return self.cov@x + rho*(x - z + u)

    def grad_z(self, x, z, u, rho, lambda_star, y_star):
        return lambda_star*self.sigma + y_star + rho*(x-z+u)

    def update_u(self, u, x, z):
        return u + (x-z)

    #Gradient Descent Method to find min x^{k+1}, fixed alpha
    def GDM_x(self, X, z, u, nb_iter, alpha, rho):
        i = 0
        while i<nb_iter:
            if not ((X - alpha*self.grad_x(x = X,z = z,u = u, rho = rho)) >=0).all():
                alpha = alpha/100
            X = X - alpha*self.grad_x(x = X,z = z,u = u, rho = rho)
            i += 1
        return X

    #Gradient Descent Method to find min z^{k+1}, fixed alpha
    def GDM_z(self, Z, x, u, nb_iter, alpha, rho, lambda_star, y_star):
        i = 0
        while i<nb_iter:
            if not ((Z - (alpha/100)*self.grad_z(x = x,z = Z,u = u, rho = rho, lambda_star = lambda_star, y_star = y_star)) >=0).all():
                alpha = alpha/100
            Z = Z - (alpha/100)*self.grad_z(x = x,z = Z,u = u, rho = rho, lambda_star = lambda_star, y_star = y_star)
            i += 1
        return Z

    def update_random_permut_ADMM(self, i, x, z, u, nb_iter_grad, alpha, rho, lambda_star, y_star):
        if i == 0:
            x = self.GDM_x(X = x, z = z, u = u, nb_iter=nb_iter_grad, alpha = alpha, rho = rho)
            return x
        elif i == 1:
            z = self.GDM_z(Z = z, x = x, u = u, nb_iter=nb_iter_grad, alpha = alpha, rho = rho, lambda_star = lambda_star, y_star = y_star)
            return z
        elif i == 2:
            u = self.update_u(u = u, x = x, z = z)
            return u

    def solve(self, nb_iter = 10000, nb_iter_grad = 500, alpha = 0.01, rho = 0.1, lambda_star = 0.1, y_star = np.random.rand(10), random_permutation = False, verbose = False):
        '''
        Implements the solver for the ADMM

        :param nb_iter: int - # of iterations of the ADMM
        :param nb_iter_grad: int - # of iterations of each Gradient Descent
        :param rho: float -  constant associated with the penalty term in the augmented Lagrangian
        :param lambda_star: float - Lagrangian multiplier
        :param y_star: np.array (10) - Lagrangian multiplier
        :param random_permutation: boolean - whether we run randomly permuted ADMM
        :param verbose: boolean - whether print or not
        '''    	
        random.seed(1515)
        i = 0
        L = []
        x = np.random.rand(10)
        x = x/np.sum(x)
        z = np.random.rand(10)
        z = z/np.sum(z)
        u = np.random.rand(10)
        if verbose:
            print('-'*80)
            print("ADMM Solver")
            print("Inialization of x: \n",x)
            print("Inialization of z: \n",z)
            print("Inialization of u: \n",u)
            print('-'*80)
        if random_permutation:
            L = [x,z,u]
        for i in tqdm(range(nb_iter)):
            if random_permutation:
                K = np.random.permutation(3)
                if not (x >= 0).all():
                    x = x + np.min(x) + 1
                if not (z >= 0).all():
                    z = z + np.min(z) + 1
                for index in K:
                    L[index] = self.update_random_permut_ADMM(i = index, x = L[0], z = L[1], u = L[2], nb_iter_grad = nb_iter_grad, alpha = alpha, rho = rho, lambda_star = lambda_star, y_star = y_star)
            else:
                if not (x >= 0).all():
                    x = x + np.min(x) + 1
                x = self.GDM_x(X = x, z = z, u = u, nb_iter=nb_iter_grad, alpha = alpha, rho = rho)

                if not (z >= 0).all():
                    z = z + np.min(z) + 1
                z = self.GDM_z(Z = z, x = x, u = u, nb_iter=nb_iter_grad, alpha = alpha, rho = rho, lambda_star = lambda_star, y_star = y_star)

                u = self.update_u(u = u, x = x, z = z)

                i+=1
                x = x/np.sum(x)
                z = z/np.sum(z)
        return x,z

if __name__ == '__main__':
    file = "./data/histo.csv"
    data = pd.read_csv(file, index_col = 0, parse_dates = True, dayfirst = False)
    admm = ADMM_Solver(data)
    admm.get_cov(by_hand = False, window = 120, date = pd.Timestamp('2019-2-28'))
    print(admm.solve(nb_iter = 10000, nb_iter_grad = 100, alpha = 0.0001, rho = 0.0001, random_permutation = True, lambda_star = 0.001, verbose = False))