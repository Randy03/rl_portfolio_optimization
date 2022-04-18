import tensorflow as tf
import numpy as np


def expand_dims(x):
    exp = tf.expand_dims(x, axis=1)
    exp = tf.expand_dims(exp, axis=1)
    exp = tf.expand_dims(exp, axis=0)
    return exp

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
        #aca va a haber que hacer el reshape del obs para que quede (features,assets,periods)
        _obs = tf.expand_dims(obs['data'],axis=0)
        _weights = obs['weights']
        _bias = _weights[0]
        _weights = _weights[1:]
        x = self.layer_1(_obs)
        x = self.layer_2(x)
        _weights = tf.keras.layers.Lambda(expand_dims)(_weights)
        x = tf.concat([x,_weights],3)
        x = self.layer_3(x)
        _bias = tf.keras.layers.Lambda(expand_dims)([_bias])
        x = tf.concat([_bias,x],1)
        x = tf.keras.layers.Flatten()(x)
        x = tf.keras.layers.Activation('softmax')(x)
        return x
    
