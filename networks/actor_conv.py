import tensorflow as tf
import numpy as np
from .functions import expand_dims,custom_activation



class Actor(tf.keras.Model):
    def __init__(self,state_dim,action_dim,max_action):
        super(Actor,self).__init__()
        self.layer_1 = tf.keras.layers.Conv2D(filters=2, kernel_size=(1,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        #self.layer_2 = tf.keras.layers.Conv2D(filters=20, kernel_size=(1,48), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.layer_2 = tf.keras.layers.Conv2D(filters=20, kernel_size=(1,10), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.layer_2_ = tf.keras.layers.Conv2D(filters=60, kernel_size=(1,30), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.layer_2__ = tf.keras.layers.Conv2D(filters=10, kernel_size=(1,10), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.layer_3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.max_action = max_action
        
    def call(self, obs):
        _obs = obs["data"]
        _bias = tf.keras.layers.Lambda(expand_dims)(obs["weights"][:,0].reshape(-1,1))
        _weights = tf.keras.layers.Lambda(expand_dims)(obs["weights"][:,1:])
        x = self.layer_1(_obs)
        x = self.layer_2(x)
        x = self.layer_2_(x)
        x = self.layer_2__(x)
        x = tf.concat([x,_weights],3)
        x = self.layer_3(x)
        x = tf.concat([_bias,x],1)
        x = tf.keras.layers.Flatten()(x)
        #x = tf.keras.layers.Activation(custom_activation)(x)
        x = tf.keras.activations.softmax(x)
        return x
    
