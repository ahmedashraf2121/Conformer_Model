import math

import numpy as np
import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa


class MHSA(keras.Model):
    def __init__(self,batch_size,**kwargs):
        super().__init__(**kwargs)
        self.LayerNorm = keras.layers.LayerNormalization(axis=1)
        self.MultiHeadAttention = tfa.layers.MultiHeadAttention(head_size=16,num_heads=4)
        self.dropout = keras.layers.Dropout(0.1)
        self.add = keras.layers.Add()
        self.batch_size = batch_size

    def call(self,inputs):
        positional_embedding = np.array([math.sin(i/144) for i in range(144)])[np.newaxis,:]
        x = self.add([inputs,positional_embedding])
        x = self.LayerNorm(x)
        x = self.MultiHeadAttention([x,x,x])
        x = self.dropout(x)
        x = seld.add([x,inputs])
        return x
