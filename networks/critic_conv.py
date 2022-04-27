import tensorflow as tf

def expandDims(x):
    expX = tf.expand_dims(x, axis=-1)
    expX = tf.expand_dims(expX, axis=-1)
    return expX

class Critic(tf.keras.Model):
    def __init__(self,state_dim,action_dim):
        super(Critic,self).__init__()
        self.layer_1 = tf.keras.layers.Conv2D(filters=2, kernel_size=(1,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.layer_2 = tf.keras.layers.Conv2D(filters=20, kernel_size=(1,48), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.layer_3 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.layer_4 = tf.keras.layers.Dense(11, activation='relu')
        self.layer_5 = tf.keras.layers.Dense(1, activation='relu')
        
        self.layer_6 = tf.keras.layers.Conv2D(filters=2, kernel_size=(1,3), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.layer_7 = tf.keras.layers.Conv2D(filters=20, kernel_size=(1,48), activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.layer_8 = tf.keras.layers.Conv2D(filters=1, kernel_size=(1,1), kernel_regularizer=tf.keras.regularizers.l2(0.001))
        self.layer_9 = tf.keras.layers.Dense(11, activation='relu')
        self.layer_10 = tf.keras.layers.Dense(1, activation='relu')
        
    def call(self, obs,actions):
        _obs = obs["data"]
        _bias = tf.keras.layers.Lambda(expand_dims)(obs["weights"][:,0].reshape(-1,1))
        _weights = tf.keras.layers.Lambda(expand_dims)(obs["weights"][:,1:])
        x1 = self.layer_1(_obs)
        x1 = self.layer_2(x1)
        x1 = tf.concat([x1,_weights],3)
        x1 = self.layer_3(x1)
        x1 = tf.concat([_bias,x1],1)
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.Dropout(0.2)(x1)
        x1 = self.layer_4(x1)
        x1 = tf.concat([actions,x1],1)
        x1 = self.layer_5(x1)
        
        x2 = self.layer_6(_obs)
        x2 = self.layer_7(x2)
        x2 = tf.concat([x2,_weights],3)
        x2 = self.layer_8(x2)
        x2 = tf.concat([_bias,x2],1)
        x2 = tf.keras.layers.Flatten()(x2)
        x2 = tf.keras.layers.Dropout(0.2)(x2)
        x2 = self.layer_9(x2)
        x2 = tf.concat([actions,x2],1)
        x2 = self.layer_10(x2)
        
        return x1, x2
        
    def Q1(self, state, action):
        _obs = obs["data"]
        _bias = tf.keras.layers.Lambda(expand_dims)(obs["weights"][:,0].reshape(-1,1))
        _weights = tf.keras.layers.Lambda(expand_dims)(obs["weights"][:,1:])
        x1 = self.layer_1(_obs)
        x1 = self.layer_2(x1)
        x1 = tf.concat([x1,_weights],3)
        x1 = self.layer_3(x1)
        x1 = tf.concat([_bias,x1],1)
        x1 = tf.keras.layers.Flatten()(x1)
        x1 = tf.keras.layers.Dropout(0.2)(x1)
        x1 = self.layer_4(x1)
        x1 = tf.concat([actions,x1],1)
        x1 = self.layer_5(x1)
        return x1