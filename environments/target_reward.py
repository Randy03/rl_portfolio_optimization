from .base import PortfolioEnvironment
import numpy as np

class PortfolioEnvironmentTargetReward(PortfolioEnvironment):
    def __init__(self,assets_names_list,assets_data_list,target,fee,initial_capital=100000,look_back_window=50,max_steps=200):
        super(PortfolioEnvironmentTargetReward,self).__init__(assets_names_list,assets_data_list,fee,initial_capital,look_back_window,max_steps)
        self.target = target
        self.target_amount = self.initial_capital / self.target[self.buffer.pointer]
        
    def _calculate_reward(self, p1):
        return (p1 * self.initial_capital - self.target_amount * self.target[self.buffer.pointer])/self.max_steps
    
    def reset(self):
        self.weights = np.array([1.0]+[0.0]*(self.m))
        self.portfolio_value = 1.0
        self.portfolio_value_units = self.initial_capital * self.portfolio_value
        self.current_step = 0
        self.buffer.reset()
        self.target_amount = self.initial_capital / self.target[self.buffer.pointer]
        return {"data":self.buffer.get_batch(normalize=self.normalize),"weights":self.weights}