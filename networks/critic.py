import tensorflow as tf

class Critic(tf.keras.Model):
    def __init__(self,state_dim,action_dim):
        super(Critic,self).__init__()
        self.layer_1 = tf.keras.layers.Dense(state_dim+action_dim,activation='relu',kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1./3., distribution = 'uniform'))
        self.layer_2 = tf.keras.layers.Dense(400,activation='relu',kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1./3., distribution = 'uniform'))
        self.layer_3 = tf.keras.layers.Dense(300,activation='relu',kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1./3., distribution = 'uniform'))
        self.layer_4 = tf.keras.layers.Dense(1)
        self.layer_5 = tf.keras.layers.Dense(state_dim+action_dim,activation='relu',kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1./3., distribution = 'uniform'))
        self.layer_6 = tf.keras.layers.Dense(400,activation='relu',kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1./3., distribution = 'uniform'))
        self.layer_7 = tf.keras.layers.Dense(300,activation='relu',kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1./3., distribution = 'uniform'))
        self.layer_8 = tf.keras.layers.Dense(1)
        
    def call(self, obs,actions):
        x0 = tf.concat([obs, actions], 1)
        x1 = self.layer_1(x0)
        x1 = self.layer_2(x1)
        x1 = self.layer_3(x1)
        x1 = self.layer_4(x1)
        
        x2 = self.layer_5(x0)
        x2 = self.layer_6(x2)
        x2 = self.layer_7(x2)
        x2 = self.layer_8(x2)
        
        return x1, x2
        
    def Q1(self, state, action):
        x0 = tf.concat([state, action], 1)
        x1 = self.layer_1(x0)
        x1 = self.layer_2(x1)
        x1 = self.layer_3(x1)
        x1 = self.layer_4(x1)
        return x1