import pandas as pd
import numpy as np

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

class backtester:
    
    def __init__(self, data):
        self.data = data.ffill()
        self.res = pd.DataFrame(data = [], index = data.index)
        
    def interpol():
        pass
        
    def get_data(self):
        self.ret = self.data/self.data.shift(1) - 1
        self.log_ret = np.log(self.data/self.data.shift(1))
    
    def rebalancing(self, freq = "Monthly", method = "inv vol"):
        if freq == "Monthly":
            datetime_mask = (pd.DatetimeIndex(np.roll(data.index.values,1)).month != data.index.month)
            
        if method == "inv vol":
            self.inv_vol_weights(datetime_mask)
        
    def inv_vol_weights(self, datetime_mask, window = 120):
        self.vols = np.sqrt( 252 / window * np.square(self.log_ret).rolling(window).sum() )
        self.weights = 1/self.vols[datetime_mask].reindex(self.data.index)
        self.weights=(self.weights-self.weights.mean())/self.weights.std()
        self.weights = self.weights.ffill()
    
    def MDP_weights(self):
        pass
    
    #BT = sum of the daily returns == Profits not reinvested
    def pnl(self):
        self.get_data()
        self.rebalancing()
        self.pnl = self.weights * self.ret
        self.pnl = self.pnl.dropna()
        self.pnl = self.pnl.sum(axis = 1)
        self.pnl = self.pnl.cumsum()
        pd.DataFrame(self.pnl).to_clipboard()
    

if __name__ == "__main__":
    path = "C:\\Users\\Raphael\\Documents\\Stanford2018\\Q2\\CME307\\Project\\git\\data\\"
    filename = "histo.csv"
    data = pd.read_csv(path + filename, index_col = 0, parse_dates = True, dayfirst = False)
    BT = backtester(data)
    BT.pnl()
    