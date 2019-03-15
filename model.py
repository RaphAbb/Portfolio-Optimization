import numpy as np
import pandas as pd
from tqdm import tqdm
import random
from datetime import datetime

class ADMM_Solver():

    def __init__(self, data):
        #self.weights = np.random.rand(10,1)
        self.data = data
        self.ret = self.data/self.data.shift(1)-1
        self.log_ret = np.log(self.data/self.data.shift(1)).dropna()

    def cov_matrix(self, date, window = 120):
        data_ = self.data.loc[date:]
        data_ = data_[:window]
        return np.array(data_.cov())

    def cov_matrix_hand(self, date, window = 120):
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

    def solve(self, nb_iter = 10000, nb_iter_grad = 500, alpha = 0.01, rho = 0.1, lambda_star = 0.1, y_star = np.random.rand(10)):
        random.seed(1515)
        i = 0
        L = []
        x = np.random.rand(10)
        x = x/np.sum(x)
        z = np.random.rand(10)
        z = z/np.sum(z)
        u = np.random.rand(10)
#        print('-'*80)
#        print("ADMM Solver")
#        print("Inialization of x: \n",x)
#        print("Inialization of z: \n",z)
#        print("Inialization of u: \n",u)
#        print('-'*80)
        for i in tqdm(range(nb_iter)):
            if not (x >= 0).all():
                #print("This is x: \n", x)
                x = x + np.min(x) + 1
                #print(i)
            x = self.GDM_x(X = x, z = z, u = u, nb_iter=nb_iter_grad, alpha = alpha, rho = rho)
            if not (z >= 0).all():
                #print("This is z: \n", z)
                z = z + np.min(z) + 1
                #print("This is z: \n", z)
                #print(i)
            #print("This is x: \n", x)
            z = self.GDM_z(Z = z, x = x, u = u, nb_iter=nb_iter_grad, alpha = alpha, rho = rho, lambda_star = lambda_star, y_star = y_star)
            #print("This is z: \n", z)
            u = self.update_u(u = u, x = x, z = z)
            #print("This is u: \n", u)
            i+=1
            x = x/np.sum(x)
            z = z/np.sum(z)
        return x,z

if __name__ == '__main__':
    file = "./data/histo.csv"
    data = pd.read_csv(file, index_col = 0, parse_dates = True, dayfirst = False)
    admm = ADMM_Solver(data)
    admm.get_cov(by_hand = False, window = 120, date = pd.Timestamp('2019-2-28'))
    print(admm.solve(nb_iter = 10000, nb_iter_grad = 100, alpha = 0.0001, rho = 0.0001, lambda_star = 0.001))