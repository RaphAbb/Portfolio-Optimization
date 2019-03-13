import numpy as np
import pandas as pd
from utils import data_to_pd, cov_matrix_hand, cov_matrix_pd

class ADMM_Solver():

    def __init__(self, path, window, date, nb_iter, beta):
        self.path = path
        self.window = window
        self.cov = np.zeros([10,10])
        self.date = date
        self.weights = np.random.rand(10,1)
        self.lambda_ = np.random.rand(2,1)

    def get_data(self, path):
        self.data = pd.read_excel(path, sheet_name = 'data_post')
        self.data = self.data.set_index('date').dropna()
        self.ret = self.data/self.data.shift(1)-1
        self.log_ret = np.log(self.data/self.data.shift(1))
        self.A = np.array([self.log_ret,np.ones(10)])
'''
TO BE UPDATED
    def cov_matrix(self, date, window = 120, drop_backtest = True):
        if drop_backtest:
            data = data.drop(['MSCI'], axis = 1)
        return np.array(data.cov())

    def cov_matrix_hand(data, window = 120, drop_backtest = True):
        if drop_backtest:
            data = data.drop(['MSCI'], axis = 1)
        A = np.zeros([10,10])
        n = data.shape[0]
        sum_ = np.array(data.sum(axis = 0))
        y_bar = sum_/n
        index_ = {0:'SPY',1:'VTI',2:'BND',3:'EMB',4:'TLT',5:'MBB',6:'IAU',7:'USO',8:'USDEUR',9:'USDJPY'}
        for i in range(10):
            for j in range(10):
                if i == j:
                    A[(i,i)] = np.sum(np.square(data[index_[i]] - y_bar[i]))/n
                else:
                    A[(i,j)] = np.sum((data[index_[i]] - y_bar[i])*(data[index_[j]] - y_bar[j]))/n
        return A
'''
    def get_cov(self, by_hand, window, date)
        if by_hand:
            return cov_matrix_hand(self.data)
        else:
        	return cov_matrix_pd(self.data)

    def divide_A(self):
        return [self.A[i:i+5] for i in range(0,10,5)]

    def divide_weights(self):
        return [self.weights[i:i+5] for i in range(0,10,5)]

    def update_l(self, l, beta):
    	A = self.divide_A()
    	w = self.divide_weights()
    	L = np.zeros([2,1])
    	for i  in range(2):
    		L += A[i]@w[i]
    	self.lambda_ = self.lambda_ + self.beta*(L - np.array([[1],[1]]))
    	return self.lambda_

    def lc(self):
    	L = np.zeros([2,1])
    	for i  in range(2):
    		L += A[i]@w[i]
        lc = np.transpose(self.weights)@self.cov@self.weights/2 + self.lambda_@L + (c/2)*np.transpose(L)@L
        return lc
