import numpy as np
    
    
class ReplayBufferMultiObs():
    def __init__(self,max_size=1e6):
        self.max_size = max_size
        self.storage = []
        self.pointer = 0 
    
    def add(self, transition):
        if len(self.storage)==self.max_size:
            self.storage[int(self.pointer)] = transition
            self.pointer = (self.pointer+1)%self.max_size
        else:
            self.storage.append(transition)
    
    def sample(self,batch_size):
        ind = np.random.randint(0,len(self.storage),size=batch_size) 
        batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = {"data":[],"weights":[]},{"data":[],"weights":[]},[],[],[]
        for i in ind:
            state, next_state, action, reward, done = self.storage[i]
            batch_states["data"].append(np.array(state["data"],copy=False))
            batch_states["weights"].append(np.array(state["weights"],copy=False))
            batch_next_states["data"].append(np.array(next_state["data"],copy=False))
            batch_next_states["weights"].append(np.array(next_state["weights"],copy=False))
            batch_actions.append(np.array(action,copy=False))
            batch_rewards.append(np.array(reward,copy=False))
            batch_dones.append(np.array(done,copy=False))
        batch_states["data"] = np.array(batch_states["data"])
        batch_states["weights"] = np.array(batch_states["weights"])
        batch_next_states["data"] = np.array(batch_next_states["data"])
        batch_next_states["weights"] = np.array(batch_next_states["weights"])
        
        return batch_states,batch_next_states,np.array(batch_actions),np.array(batch_rewards).reshape(-1,1),np.array(batch_dones).reshape(-1,1)
