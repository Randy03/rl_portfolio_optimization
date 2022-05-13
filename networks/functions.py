import numpy as np
import tensorflow as tf

def expand_dims(x):
    expX = tf.expand_dims(x, axis=-1)
    expX = tf.expand_dims(expX, axis=-1)
    return expX

def custom_activation(x):
    tensor = tf.clip_by_value(x, clip_value_min=0, clip_value_max=100)
    return tensor/tf.reduce_sum(tensor)
