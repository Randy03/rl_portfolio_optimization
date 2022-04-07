import torch.nn.functional as F
import torch


class Critic(torch.nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic,self).__init__() #Como los modelos criticos van de a pares tengo q repetir las layers, cada gemelo tiene 3 layers
        self.layer_1 = torch.nn.Linear(state_dim+action_dim,400)
        self.layer_2 = torch.nn.Linear(400,300)
        self.layer_3 = torch.nn.Linear(300,1)
        self.layer_4 = torch.nn.Linear(state_dim+action_dim,400)
        self.layer_5 = torch.nn.Linear(400,300)
        self.layer_6 = torch.nn.Linear(300,1)
    
    def forward(self,x,u): #x:stados , u:acciones
        xu = torch.cat([x,u], 1) #concatenar verticalmente los estados con acciones
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        x2 = F.relu(self.layer_4(xu))
        x2 = F.relu(self.layer_5(x2))
        x2 = self.layer_6(x2)
        return x1,x2
    
    def Q1(self,x,u): #x:stados , u:acciones forwardear solo uno de los criticos
        xu = torch.cat([x,u], 1) #concatenar verticalmente los estados con acciones
        x1 = F.relu(self.layer_1(xu))
        x1 = F.relu(self.layer_2(x1))
        x1 = self.layer_3(x1)
        return x1
    