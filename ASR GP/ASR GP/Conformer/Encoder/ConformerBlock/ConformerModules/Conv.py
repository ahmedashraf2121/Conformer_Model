import tensorflow as tf
from tensorflow import keras

class Conv_module(keras.Model):
    def __init__(self,encoder_width,**kwargs):
        super().__init__(**kwargs)
        self.LayerNorm = keras.layers.LayerNormalization(axis=1)
        self.Conv1D1 = keras.layers.Conv1D(filters=1,kernel_size=1,input_shape=(encoder_width,1))
        self.BatchNorm = keras.layers.BatchNormalization()
        self.swish = keras.activations.swish
        self.Conv1D2 = keras.layers.Conv1D(1,1)
        self.DropOut = keras.layers.Dropout(0.1)
        self.Add = keras.layers.Add()

    def call(self, inputs):
        inputs = tf.expand_dims(inputs,axis=-1)
        x = self.LayerNorm(inputs)
        x = self.Conv1D1(x)
        x = self.BatchNorm(x)
        x = self.swish(x)
        x = self.Conv1D2(x)
        x = self.DropOut(x)
        x = self.Add([x,inputs])
        x = tf.squeeze(x,axis=-1)
        return x
