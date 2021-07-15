import tensorflow as tf
from tensorflow import keras

class FFN(keras.Model):
    def __init__(self,encoder_width,**kwargs):
        super().__init__(**kwargs)
        self.LayerNorm = keras.layers.LayerNormalization(axis=1)
        self.Dense1 = keras.layers.Dense(4*encoder_width)
        self.Swish = keras.activations.swish
        self.DropOut1 = keras.layers.Dropout(0.1)
        self.Desnse2 = keras.layers.Dense(encoder_width)
        self.DropOut2 = keras.layers.Dropout(0.1)
        self.Add = keras.layers.Add()

    def call(self, inputs):
        x=self.LayerNorm(inputs)
        x=self.Dense1(x)
        x=self.Swish(x)
        x=self.DropOut1(x)
        x=self.Desnse2(x)
        x=self.DropOut2(x)
        x=0.5*x
        x=self.Add([x,inputs])
        return x
