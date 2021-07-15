


from ConformerBlock import Conformer
import tensorflow as tf
from tensorflow import keras

class Encoder(keras.Model):
    def __init__(self,num_conformers,head_size,num_heads,encoder_width,**kwargs):
        super().__init__(**kwargs)
        #-------------the block for convolution subsampling------------------
        self.conv1=keras.layers.Conv2D(1,(2,2),strides=2,input_shape=(111,384,80,1))
        self.relu1=keras.activations.relu
        self.conv2=keras.layers.Conv2D(1,(2,2),strides=2)
        self.relu2=keras.activations.relu
        ----------------------------------------------------------------
        self.flat=keras.layers.Flatten()
        self.linear=keras.layers.Dense(encoder_width)
        self.dropout=keras.layers.Dropout(0.1)
        #------------------the conformer blocks stacked together------------------------
        self.conformers=[Conformer_block(head_size,num_heads,encoder_width) for i in range(num_conformers)]
    def call(self,inputs,training):
        x=self.conv1(inputs)
        x=self.relu1(x)
        x=self.conv2(x)
        x=self.relu2(x)
        x=self.flat(x)
        x=self.linear(x)
        x=self.dropout(x)
        for layer in self.conformers:
            x=layer(x)
        return x

