import pandas as pd
import numpy as np

import model as model

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class backtester:
    
    def __init__(self, data, freq = "Monthly", method = "inv vol"):
        self.data = data.ffill()
        self.res = pd.DataFrame(data = [], index = data.index)
        self.freq = freq
        self.method = method
        
    def interpol():
        pass
        
    def get_data(self):
        self.ret = self.data/self.data.shift(1) - 1
        self.log_ret = np.log(self.data/self.data.shift(1))
    
    #Method computing weights of the portfolio
    #for a given methodology, for a given frequency of weight rebalancing
    def rebalancing(self, freq, method):
        if freq == "Monthly":
            datetime_mask = (pd.DatetimeIndex(np.roll(self.data.index.values,1)).month != self.data.index.month)
            
        if method == "inv vol":
            self.inv_vol_weights(datetime_mask)
        elif method == "MDP Short Selling":
            self.MDP_weights(datetime_mask)
        elif method == "MDP":
            self.MDP_pos_weights(datetime_mask)
        
    #Weights are attributed in 1/(volatility of the asset)
    def inv_vol_weights(self, datetime_mask, window = 120):
        self.vols = np.sqrt( 252 / window * np.square(self.log_ret).rolling(window).sum() )
        self.weights = 1/self.vols[datetime_mask].reindex(self.data.index)
        self.weights = self.weights.div(self.weights.sum(axis=1), axis=0)
        self.weights = self.weights.ffill()
    
    #Weights are attributed according to the Most Diversified Portfolio Methodology, authorising short selling
    def MDP_weights(self, datetime_mask, window = 120):
        #change the -1 in time window
        self.vols = np.sqrt( 252 / window * np.square(self.log_ret).rolling(window-1).sum() )
        self.weights = pd.DataFrame(data = [], index = self.data[datetime_mask].index, columns = self.data.columns)
        
        #start dates and end dates used in comatrice computation
        t_start = pd.DatetimeIndex(np.roll(self.data.index.values, window))[datetime_mask].dropna()
        t_end = self.data.index[datetime_mask].dropna()
        for i, date in enumerate(t_end):
            if t_end[i] < t_start[i]:
                pass
            else:
                temp_data = self.log_ret[(self.log_ret.index <= t_end[i]) * (self.log_ret.index >= t_start[i])]
                cov = temp_data.cov()
                vol = self.vols[self.vols.index == date]
                invcov = pd.DataFrame(np.linalg.pinv(cov.values), cov .columns, cov .index)
                self.weights[self.weights.index == date] = (invcov.dot(vol.transpose()) /( vol.dot(invcov.dot(vol.transpose())) ).values[0][0]).transpose()
        self.weights = self.weights.dropna()
        self.weights = self.weights.div(self.weights.sum(axis=1), axis=0)
        self.weights = self.weights.reindex(self.data.index)
        self.weights = self.weights.ffill()

    #Weights are attributed according to the Most Diversified Portfolio Methodology, without short selling
    def MDP_pos_weights(self, datetime_mask, window = 120):
        #change the -1 in time window
        self.vols = np.sqrt( 252 / window * np.square(self.log_ret).rolling(window-1).sum() )
        self.weights = pd.DataFrame(data = [], index = self.data[datetime_mask].index, columns = self.data.columns)
        
        #start dates and end dates used in comatrice computation
        t_start = pd.DatetimeIndex(np.roll(self.data.index.values, window))[datetime_mask].dropna()
        t_end = self.data.index[datetime_mask].dropna()
        for i, date in enumerate(t_end):
            if t_end[i] < t_start[i]:
                pass
            else:
                ADMM = model.ADMM_Solver(self.data)
                ADMM.get_cov(by_hand = False, window = window, date = date)
                x,y = ADMM.solve(nb_iter = 10000, nb_iter_grad = 100, alpha = 0.0001, rho = 0.0001, lambda_star = 0.001)
                self.weights.loc[i] = (x+y)/2
        self.weights = self.weights.dropna()
        self.weights = self.weights.div(self.weights.sum(axis=1), axis=0)
        self.weights = self.weights.reindex(self.data.index)
        self.weights = self.weights.ffill()
        
    #BT = sum of the daily returns == Profits not reinvested
    def pnl(self):
        self.get_data()
        self.rebalancing(self.freq, self.method)
        self.pnl = self.weights.shift(1) * self.ret
        self.pnl = self.pnl.dropna()
        self.pnl = self.pnl.sum(axis = 1)
        self.pnl = self.pnl.cumsum()
        pd.DataFrame(self.pnl).to_clipboard()
    

if __name__ == "__main__":
    file = "./data/histo.csv"
    data = pd.read_csv(file, index_col = 0, parse_dates = True, dayfirst = False)
    BT1 = backtester(data, freq = "Monthly", method = "inv vol")
    BT2 = backtester(data, freq = "Monthly", method = "MDP Short Selling")
    BT3 = backtester(data, freq = "Monthly", method = "MDP")
    BT1.pnl()
    BT2.pnl()
    BT3.pnl()
    

####Comparaisons
    BT1.pnl.rolling(252).mean() / BT1.pnl.rolling(252).std() / np.sqrt(252)
    BT2.pnl.rolling(252).mean() / BT2.pnl.rolling(252).std() / np.sqrt(252)
    BT1.pnl.rolling(504).mean() / BT1.pnl.rolling(504).std() / np.sqrt(252)
    BT2.pnl.rolling(504).mean() / BT2.pnl.rolling(504).std() / np.sqrt(252)
    BT1.pnl.rolling(252*5).mean() / BT1.pnl.rolling(252*5).std() / np.sqrt(252)
    BT2.pnl.rolling(252*5).mean() / BT2.pnl.rolling(252*5).std() / np.sqrt(252)
    