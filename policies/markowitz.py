from scipy.optimize import minimize
import numpy as np

class MarkowitzAgent():
    def __init__(self,rf=0.01):
        self.rf= rf
        self.data = None
    
    def select_action(self,state):
        self.data = state["data"][:,:,0].T
        length = self.data.shape[1]+1
        cons = ({'type':'eq','fun':self._sum_of_weights_check})
        bounds = tuple(zip([0],[0.05])) + tuple(zip([0]*(length-1),[1]*(length-1)))
        start_weights = [1/length] * (length)
        return minimize(self._minimize_function,start_weights,method='SLSQP',bounds=bounds,constraints=cons).x
    
    def _stock_return_volatility(self,data,weights):
        weights = np.array(weights)
        log_ret = np.log(data[1:]/data[:-1])
        log_ret = np.concatenate((np.ones(shape=(log_ret.shape[0],1)),log_ret),axis=1)
        n_periods = len(data)
        ret = np.sum((np.mean(log_ret,axis=0)*weights)*n_periods)
        vol = np.sqrt(np.dot(weights,np.dot(np.cov(log_ret.T)*n_periods,weights.T)))
        SR = (ret-self.rf)/vol
        return SR
    
    def _minimize_function(self,weights):
        return self._stock_return_volatility(data=self.data,weights=weights)*-1
    
    def _sum_of_weights_check(self,weights):
        return np.sum(weights)-1
    