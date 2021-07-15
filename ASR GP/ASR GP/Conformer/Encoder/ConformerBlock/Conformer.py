import tensorflow as tf
from tensorflow import keras
from ConformerModules.FFN import FFN
from ConformerModules.MHSA import MHSA
from ConformerModules.Conv import Conv

class Conformer_block(keras.Model):
    def __init__(self,head_size,num_heads,encoder_width,**kwargs):
        super().__init__(**kwargs)
        self.ff1 = FFN(encoder_width)
        self.multihead = MHSA(head_size,num_heads)
        self.conv = Conv(encoder_width)
        self.ff2 = FFN(encoder_width)

    def call(self,inputs):
        x = self.ff1(inputs)
        x = self.multihead(x)
        x = self.conv(x)
        x = self.ff2(x)
        return x
