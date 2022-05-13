import tensorflow as tf
from networks import ActorConv as Actor, CriticConv as Critic
import datetime

from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

#def reshape_state(state):
#    if isinstance(state,dict):
#        for key,value in state.items():
#            state[key] = value.reshape(1,-1)
#    else:
#        state = state.reshape(1,-1)
#    return state

class TD3():
    def __init__(self, state_dim, action_dim, max_action,lr=3e-3):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        for t, e in zip(self.actor_target.trainable_variables, self.actor.trainable_variables):
            t.assign(e)
        
        self.actor_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        
        self.critic = Critic(state_dim, action_dim)
        self.critic_target = Critic(state_dim, action_dim)
        for t, e in zip(self.critic_target.trainable_variables, self.critic.trainable_variables):
            t.assign(e)
        self.critic_optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
        self.critic_loss_fn = tf.keras.losses.Huber()
        self.max_action = max_action
        
    def select_action(self, state):
        #state = reshape_state(state)
        action = self.actor.call(state)[0].numpy()
        return action
    
    def train(self, replay_buffer, iterations, batch_size=100, discount=0.99, tau=0.005, policy_noise=0.2, noise_clipping=0.5, policy_freq=2):
        for i in range(iterations):
            #get sample (s,s',a,r) from memory
            batch_states, batch_next_states, batch_actions, batch_rewards, batch_dones = replay_buffer.sample(batch_size)
            
            #predict a' from s'
            next_action = self.actor_target.call(batch_next_states)
            
            #add noise 
            noise = tf.random.normal(next_action.shape, mean=0, stddev=policy_noise)
            noise = tf.clip_by_value(noise, -noise_clipping, noise_clipping)
            #next_action = tf.clip_by_value(next_action + noise, -self.max_action, self.max_action)
            next_action = tf.nn.softmax(next_action + noise).numpy()
                        
            target_Q1,target_Q2 = self.critic_target.call(batch_next_states,next_action)
            #take minimum Q value
            target_Q = tf.minimum(target_Q1,target_Q2)
            #get final Q target, considering wether the episode has ended or not
            target_Q = tf.stop_gradient(batch_rewards + (1 - batch_dones) * discount * target_Q)    
            
            #critic backpropagation
            
            trainable_critic_variables = self.critic.trainable_variables
            
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(trainable_critic_variables)
                current_Q1, current_Q2 = self.critic.call(batch_states,batch_actions)
                critic_loss = (self.critic_loss_fn(current_Q1,target_Q) + self.critic_loss_fn(current_Q2,target_Q))
            critic_grads = tape.gradient(critic_loss, trainable_critic_variables)
            self.critic_optimizer.apply_gradients(zip(critic_grads, trainable_critic_variables))
                     
            #AUpdate actor model
            if i%policy_freq==0:
                trainable_actor_variables = self.actor.trainable_variables
                with tf.GradientTape(watch_accessed_variables=False) as tape:
                    tape.watch(trainable_actor_variables)
                    #applying gradient ascent by taking de oposit function
                    actor_loss = -tf.reduce_mean(self.critic.Q1(batch_states, self.actor(batch_states))) 
                actor_grads = tape.gradient(actor_loss, trainable_actor_variables)
                self.actor_optimizer.apply_gradients(zip(actor_grads, trainable_actor_variables))
            
                # update the weights in the critic and actor target models, the tau parameter will define how much is going to adjust
                for target_param, param in zip(self.critic_target.trainable_variables, self.critic.trainable_variables):
                    target_param.assign(target_param * (1 - tau) + param * tau)
                for target_param, param in zip(self.actor_target.trainable_variables, self.actor.trainable_variables):
                    target_param.assign(target_param * (1 - tau) + param * tau)
        
        
        
        
    def save(self,folder_path):
        self.actor.save_weights(f'{folder_path}/actor')
        self.actor_target.save_weights(f'{folder_path}/actor_target')
        self.critic.save_weights(f'{folder_path}/critic')
        self.critic_target.save_weights(f'{folder_path}/critic_target')
        
    def load(self,folder_path):
        self.actor.load_weights(f'{folder_path}/actor')
        self.actor_target.load_weights(f'{folder_path}/actor_target')
        self.critic.load_weights(f'{folder_path}/critic')
        self.critic_target.load_weights(f'{folder_path}/critic_target')