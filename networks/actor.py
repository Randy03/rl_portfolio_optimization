import tensorflow as tf

class Actor(tf.keras.Model):
    def __init__(self,state_dim,action_dim,max_action):
        super(Actor,self).__init__()
        self.layer_1 = tf.keras.layers.Dense(state_dim,activation='relu',kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1./3., distribution = 'uniform'))
        self.layer_2 = tf.keras.layers.Dense(400,activation='relu',kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1./3., distribution = 'uniform'))
        self.layer_3 = tf.keras.layers.Dense(300,activation='relu',kernel_initializer=tf.keras.initializers.VarianceScaling(scale=1./3., distribution = 'uniform'))
        self.layer_4 = tf.keras.layers.Dense(action_dim,activation='softmax')
        #self.layer_4 = tf.keras.layers.Dense(action_dim,activation='tanh')
        self.max_action = max_action
        
    def call(self, obs):
        x = self.layer_1(obs)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = x * self.max_action
        return x