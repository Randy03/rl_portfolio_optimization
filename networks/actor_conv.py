import tensorflow as tf
import numpy as np


#def expand_dims(x):
#    exp = tf.expand_dims(x, axis=1)
#    exp = tf.expand_dims(exp, axis=1)
#    exp = tf.expand_dims(exp, axis=0)
#    return exp

def expand_dims(x):
    expX = tf.expand_dims(x, axis=-1)
    expX = tf.expand_dims(expX, axis=-1)
    return expX

class Actor(tf.keras.Model):
    def __init__(self,state_dim,action_dim,max_action):
        super(Actor,self).__init__()
        self.layer_1 = tf.keras.layers.Conv2D(filters=2, kernel_size=(1,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.layer_2 = tf.keras.layers.Conv2D(filters=20, kernel_size=(1,48), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.layer_3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), kernel_regularizer=tf.keras.regularizers.l2(0.001))
        #self.layer_4 = tf.keras.layers.Dense(action_dim,activation='softmax')
        #self.layer_4 = tf.keras.layers.Dense(action_dim,activation='tanh')
        self.max_action = max_action
        
    def call(self, obs):
        _obs = obs["data"]
        #print(type(obs["weights"]))
        _bias = tf.keras.layers.Lambda(expand_dims)(obs["weights"][:,0].reshape(-1,1))
        _weights = tf.keras.layers.Lambda(expand_dims)(obs["weights"][:,1:])
        x = self.layer_1(_obs)
        x = self.layer_2(x)
        #_weights = tf.keras.layers.Lambda(expand_dims)(_weights)
        x = tf.concat([x,_weights],3)
        x = self.layer_3(x)
        #_bias = tf.keras.layers.Lambda(expand_dims)([_bias])
        x = tf.concat([_bias,x],1)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Activation('softmax')(x)
        return x
    
