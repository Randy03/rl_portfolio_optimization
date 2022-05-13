from .base import PortfolioEnvironment
import numpy as np

class PortfolioEnvironmentTest(PortfolioEnvironment):
    def __init__(self,assets_names_list,assets_data_list,fee,initial_capital=100000,look_back_window=50):
        super(PortfolioEnvironmentTest,self).__init__(assets_names_list,assets_data_list,fee,initial_capital,look_back_window,len(assets_data_list[0])-look_back_window-1)
        
    def reset(self):
        self.weights = np.array([1.0]+[0.0]*(self.m))
        self.portfolio_value = 1.0
        self.portfolio_value_units = self.initial_capital * self.portfolio_value
        self.current_step = 0
        self.buffer.reset(self.n)
        return {"data":self.buffer.get_batch(normalize=self.normalize),"weights":self.weights}