import numpy as np
import gym
import random


class PortfolioEnvironment(gym.Env):
    def __init__(self,assets_names_list,assets_data_list,fee,initial_capital=100000,look_back_window=50,max_steps=200):
        super(PortfolioEnvironment,self).__init__()
        '''
        assets_names_list: list with the ticker of each security
        assets_data_list: list of pandas dataframes with the data of each security, must have the same length as assets_names_list and the first column of each dataframe must have the price of the asset
        fee: porcentage of operating fee, with decimal, ie 0.1 is equal to 10% fee
        initial_capital: amount of cash at the beginning
        look_back_window: amount of periods to look back while executing a step
        steps: maximum number of possible steps
        '''
        self.buffer = PortfolioBuffer(assets_names_list,np.array(list(map(lambda x: x.to_numpy(),assets_data_list))),look_back_window)
        self.fee = fee
        self.f = self.buffer.shape[2]
        self.n = look_back_window
        self.m = self.buffer.shape[0]
        self.max_steps = max_steps
        self.current_step = 0
        self.initial_capital = initial_capital
        
        self.action_space = gym.spaces.Box(low=0.0,high=1.0,shape=(self.m+1,),dtype=np.float16)
        self.observation_space = gym.spaces.Dict({"data": gym.spaces.Box(low=0,high=1,shape=(self.m,self.n,self.f),dtype=np.float16), 
                                              "weights": gym.spaces.Box(low=0.0,high=1.0,shape=(self.m+1,),dtype=np.float16)})
        
        self.weights = np.array([1.0]+[0.0]*(self.m))
        self.portfolio_value = 1.0
        self.portfolio_value_units = self.initial_capital * self.portfolio_value
        
    def _buy(self,index,price,amount):
        raise NotImplementedError

    def _sell(self,index,price,amount):
        raise NotImplementedError
        
    def _price_relative_vector(self):
        '''
        returns a matrix with the division of each assets value by the previous one
        '''
        #el problema de que la recompensa es muy baja es que los datos estan normalizados???, lo cual esta bien para el modelo
        #pero hay que meterlos si normalizar en el vector del precio relativo???
        prices = self.buffer.get_batch()[:,:,0].T
        prices_diff = prices[1:]/prices[:-1]
        prices_diff = np.concatenate((np.ones(shape=(prices_diff.shape[0],1)),prices_diff),axis=1)
        return prices_diff
        
    def _weights_at_end_of_period(self):
        '''
        returns a vector with the weights of the portfolio after the new prices but before taking any action
        '''
        y = self._price_relative_vector()[-1]
        #y = self.buffer.get_price_relative_vector()
        return np.multiply(y,self.weights)/np.dot(y,self.weights)
    
    def _operation_cost(self,weights):
        '''
        weights: vector with the new weights provided by the actor
        returns a scalar value with the cost of doing the buy/sell operations needed to get to those weights
        '''
        w_prime = self._weights_at_end_of_period()[1:]
        return self.fee * np.sum(np.abs(weights[1:]-w_prime))
    
    def _portfolio_value_after_operation(self,weights):
        '''
        weights: vector with the new weights provided by the actor
        returns a scalar with the new value of the portfolio after doing the buy/sell operations needed to get to those weights
        '''
        c = self._operation_cost(weights)
        p0 = self.portfolio_value
        y = self._price_relative_vector()[-1]
        #y = self.buffer.get_price_relative_vector()
        w = self.weights
        #print(np.dot(y, w))
        #print(np.dot(y, w))
        return p0 * (1 - c) * np.dot(y, w)
    
    def step(self, action):
        #print(type(action))
        p1 = self._portfolio_value_after_operation(action)
        
        reward = np.log(p1/self.portfolio_value) / self.max_steps
        
        self.weights = action
        
        self.portfolio_value = p1
        
        new_portfolio_value = self.initial_capital * self.portfolio_value
        #reward = new_portfolio_value - self.portfolio_value_units
        self.portfolio_value_units = new_portfolio_value
        
        done = 0 if self.buffer.length-1 > self.buffer.pointer and self.current_step < self.max_steps and  self.portfolio_value_units > self.initial_capital * 0.2 else 1 
        if done == 1:
            print(self.weights)
        info = {"weights":self.weights,"value":self.portfolio_value,"position":self.buffer.pointer}
        self.current_step += 1
        obs = {"data":self.buffer.get_next_batch(),"weights":self.weights}
        #print(reward)
        return obs, reward, done, info
    
    def reset(self):
        self.weights = np.array([1.0]+[0.0]*(self.m))
        self.portfolio_value = 1.0
        self.portfolio_value_units = self.initial_capital * self.portfolio_value
        self.current_step = 0
        self.buffer.reset()
        return {"data":self.buffer.get_batch(),"weights":self.weights}
    
    def render(self):
        pass



class PortfolioBuffer():
    def __init__(self,assets_names_list,assets_data_list,window):
        '''
        assets_names_list: list with the ticker of each security
        assets_data_list: numpy with the data of each security,shape(n_assets,n_records,n_feaures) must have the same length as assets_names_list and the first column of each dataframe must have the price of the asset
        window: amount of periods to return in the batch
        '''
        #self.names = {0:'CASH'}
        self.names = {}
        for index,value in enumerate(assets_data_list):
            self.names[index] = value
        self.shape = assets_data_list[0].shape
        for i in assets_data_list:
            if self.shape != i.shape:
                raise Exception('Data must be of the same shape')
        if len(assets_data_list) != len(assets_names_list):
            raise Exception('The length of assets_names_list is different than the amount of assets in assets_data_list')
        self.data = assets_data_list
        self.shape = self.data.shape
        self.window = window
        self.batch_cache = None
        self.price_relative_vector_cache = None
        self.length = self.shape[1]
        self.pointer = random.randrange(self.window,self.length-self.window)
    
    def get_batch(self):
        if self.batch_cache is None:
            batch = np.zeros(shape=(self.shape[0],self.window,self.shape[2]))
            for index,data in enumerate(self.data):
                batch[index] = data[self.pointer-self.window:self.pointer]/data[self.pointer-1][0]
            self.batch_cache = batch
        return self.batch_cache
    
    def get_next_batch(self):
        self.pointer += 1
        self.batch_cache = None
        self.price_relative_vector_cache = None
        return self.get_batch()
    
    def get_current_price(self,index):
        return self.data[index][self.pointer-1][0]
    
    def get_price_relative_vector(self):
        if self.price_relative_vector_cache is None:
            batch = np.zeros(shape=(self.shape[0],2,self.shape[2]))
            for index,data in enumerate(self.data):
                batch[index] = data[self.pointer-2:self.pointer]
            prices = batch[:,:,0].T
            prices_diff = prices[1]/prices[0]
            prices_diff = np.concatenate((np.ones(shape=(1)),prices_diff))
            self.price_relative_vector_cache = prices_diff
        return self.price_relative_vector_cache
    
    def reset(self,position=None):
        if not position:
            self.pointer = random.randrange(self.window,self.length-self.window)
        else:
            self.pointer = position
        self.batch_cache = None
        self.price_relative_vector_cache = None