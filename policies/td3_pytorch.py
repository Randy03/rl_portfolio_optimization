import torch.nn.functional as F
import torch
import datetime

from networks import ActorP as Actor, CriticP as Critic

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TD3(object):
    def __init__(self, state_dim, action_dim, max_action):
        self.actor = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict()) #states_dict devuelve los weights pero mapeados con cada layer en un diccionario, hay que preentrenar el target con estos weights
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters()) #la funcion parametros devuelve los weights de la red
        
        self.critic = Critic(state_dim,action_dim).to(device)
        self.critic_target = Critic(state_dim,action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters())
        self.max_action = max_action
    
    def select_action(self,state):
        state = torch.Tensor(state.reshape(1,-1)).to(device) #formatearlo en un tensor ya que la red neuronal solo acepta tensores de pytorch
        return self.actor.forward(state).cpu().data.numpy().flatten() #no se necesita la grafica para hacer forward propagation, la variable data contiene el tensor con los resultados de la prediccion, despues se convierte a numpy y se los aplana para q salga en una dimension
    
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clipping=0.5, policy_freq=2): #cuanto mas chico el discount mas grante tiene que ser el iterations ya que necesita pas iteraciones para converger
        for i in range(iterations):
            #Paso 4 tomar la muestra (s,s',a,r) de la memoria
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            state = torch.Tensor(batch_states).to(device) #convertirlos todos en tensores para q puedan ser usados por los modelos
            next_state = torch.Tensor(batch_next_states).to(device)
            action = torch.Tensor(batch_actions).to(device)
            reward = torch.Tensor(batch_rewards).to(device)
            done = torch.Tensor(batch_dones).to(device)
            #Paso 5 predecir a' a partir de s'
            next_action = self.actor_target.forward(next_state)
            #Paso 6 Agregar ruido al next_action
            noise = torch.Tensor(batch_actions).data.normal_(0,policy_noise).to(device) #la funcion normal_ toma como parametro la media(0) y el desvio standar para aplicar el ruido a esos valores
            noise = noise.clamp(-noise_clipping,noise_clipping) #recortar entre limite inferior y superior, cualquier valor por encima del limite se ajustara al valor del limite
            next_action = (next_action + noise).clamp(-self.max_action,self.max_action) #agregar el ruido al next action y cortarlo
            next_action = torch.nn.functional.softmax(next_action,dim=0)
            #Paso 7 pasar  (s',a') a los dos Critic target
            target_Q1,target_Q2 = self.critic_target.forward(next_state,next_action)
            #Paso 8 elegir el Q minimo
            target_Q = torch.min(target_Q1,target_Q2)
            #Paso 9 Obtener el Q target final, teniendo en cuenta si termino el episodio o no
            target_Q = reward + ((1 - done) * discount * target_Q).detach()   #El done toma valor 0 y 1, si el episodio termino done es 1
            #Paso 10 pasar  (s,a) a los dos Critic Model
            current_Q1, current_Q2 = self.critic.forward(state,action)
            #Paso 11 Calcular el loss entre el target_Q y los current
            critic_loss = F.mse_loss(current_Q1,target_Q) + F.mse_loss(current_Q2,target_Q)
            #Paso 12 backpropagation de los criticos
            self.critic_optimizer.zero_grad() #pone los gradientes en 0
            critic_loss.backward()
            self.critic_optimizer.step() #actualiza los pesos de los dos criticos
            #Paso 13 Actualizar el Actor model cada una cierta cantidad de operaciones, para aplicar el gradiente ascendente se puede aplicar el gradiente descendiente sobre el opuesto de la funcion
            if i%policy_freq==0:
                actor_min_func = -self.critic.Q1(state,self.actor.forward(state)).mean() #El mean hace la sumatoria/N , el action tiene que ser en funcion de los pesos del Actor model,
                self.actor_optimizer.zero_grad()
                actor_min_func.backward()
                self.actor_optimizer.step()
                #Paso 14 actualizar parte de los weights del actor model en el actor target (el paremtro tau es justamente para definir que tanto se tiene que actualizar)
                for param, target_param in zip(self.actor.parameters(),self.actor_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1-tau)*target_param.data)

                #Paso 15 hacer lo mismo con los criticos
                for param, target_param in zip(self.critic.parameters(),self.critic_target.parameters()):
                    target_param.data.copy_(tau * param.data + (1-tau)*target_param.data)
        
    def save(self,path='./models'):
        time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        torch.save(self.actor.state_dict(),f'{path}/{time}/actor.pth')
        torch.save(self.critic.state_dict(),f'{path}/{time}/critic.pth')
        