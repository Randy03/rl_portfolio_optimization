import torch.nn.functional as F
import torch

class Actor(torch.nn.Module):
    def __init__(self, state_dim, action_dim, max_action):
        super(Actor,self).__init__()
        self.layer_1 = torch.nn.Linear(state_dim,400) #Linear genera una layer con una transformacion lineal w*x, la layer en este caso tiene 400 neuronas y esta conectada a la input layer que tiene dimension state_dim
        self.layer_2 = torch.nn.Linear(400,300)
        self.layer_3 = torch.nn.Linear(300,action_dim)
        self.max_action = max_action
        
    def forward(self, x): #x: stados ,propagar hacia adelante las layers para armar la red, y agregar las funciones de activacion
        x1 = F.relu(self.layer_1(x))
        x1 = F.relu(self.layer_2(x1))
        x1 = torch.softmax(self.layer_3(x1),dim=1) #la tangente hiperbolica devuelve valores entre -1 y 1, se multiplica por el max_action para acomodar ese rango
        return x1