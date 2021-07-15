#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
import os
import tensorflow_io as tfio
os.chdir(r'C:\Users\HH\Desktop\project')
import Language_Model
import math


# In[1]:


class Feed_forward_module(keras.Model):
    def __init__(self,encoder_width,**kwargs):
        super().__init__(**kwargs)
        self.hidden1=keras.layers.LayerNormalization(axis=1)
        self.hidden2=keras.layers.Dense(4*encoder_width)
        self.hidden3=keras.activations.swish
        self.hidden4=keras.layers.Dropout(0.1)
        self.hidden5=keras.layers.Dense(encoder_width)
        self.hidden6=keras.layers.Dropout(0.1)
        self.hidden7=keras.layers.Add()
        
    def call(self, inputs):
        x=self.hidden1(inputs)
#         print('ff layer normalization')
#         print(x.shape)
        x=self.hidden2(x)
#         print('ff dense layer')
#         print(x.shape)
        x=self.hidden3(x)
#         print('ff swish activation')
#         print(x.shape)
        x=self.hidden4(x)
#         print('ff drop out')
#         print(x.shape)
        x=self.hidden5(x)
#         print('ff dense')
#         print(x.shape)
        x=self.hidden6(x)
        x=0.5*x
#         print('ff drop out')
#         print(x.shape)
        x=self.hidden7([x,inputs])
#         print('ff add')
#         print(x.shape)
        return x


# In[6]:


class Conv_module(keras.Model):
    def __init__(self,encoder_width,**kwargs):
        super().__init__(**kwargs)
        self.hidden1=keras.layers.LayerNormalization(axis=1)
        self.hidden2=keras.layers.Conv1D(filters=1,kernel_size=1,input_shape=(encoder_width,1))
        #GLU activation
        #self.hidden4=keras.layers.Conv1D(filters=2,kernel_size=32,padding='same',groups=2)
        self.hidden5=keras.layers.BatchNormalization()
        self.hidden6=keras.activations.swish
        self.hidden7=keras.layers.Conv1D(1,1)
        self.hidden8=keras.layers.Dropout(0.1)
        self.hidden9=keras.layers.Add()
        
    def call(self, inputs):
        inputs=tf.expand_dims(inputs,axis=-1)
        x=self.hidden1(inputs)
#         print('conv layernorm')
#         print(x.shape)
        x=self.hidden2(x)
#         print('conv pointwise')
#         print(x.shape)
        #x=self.hidden3(x)
        #x=self.hidden4(x)
        #print('conv separableconv1d')
        #print(x.shape)
        x=self.hidden5(x)
#         print('conv batchnormalization')
#         print(x.shape)
        x=self.hidden6(x)
#         print('conv swish activation')
#         print(x.shape)
        x=self.hidden7(x)
#         print('conv pointwise')
#         print(x.shape)
        x=self.hidden8(x)
#         print('conv dropout')
#         print(x.shape)
        x=self.hidden9([x,inputs])
#         print('conv add')
#         print(x.shape)
        x=tf.squeeze(x,axis=-1)
        return x


# In[ ]:


#class Multi_Head_Attention(keras.Model):
    #def __init__(self,head_size,num_heads,**kwargs):
        #super().__init__(**kwargs)
        #self.


# In[4]:


class Multihead_module(keras.Model):
    def __init__(self,head_size,num_heads,**kwargs):
        super().__init__(**kwargs)
        self.layernorm=keras.layers.LayerNormalization(axis=1)
        self.multihead=tfa.layers.MultiHeadAttention(head_size,num_heads)
        self.dropout=keras.layers.Dropout(0.1)
        self.add=keras.layers.Add()
        
    def call(self,inputs):
        positional_embedding=np.array([math.sin(i/144) for i in range(144)])[np.newaxis,:]
        x=inputs+positional_embedding
        x=self.layernorm(x)
#         print('multihead layernorm')
#         print(x.shape)
        x=self.multihead([x,x,x])
#         print('multihead multihead')
#         print(x.shape)
        x=self.dropout(x)
#         print('multihead drop out')
#         print(x.shape)
        x=self.add([x,inputs])
#         print('multihead add')
#         print(x.shape)
        return x


# In[8]:


class Conformer_block(keras.Model):
    def __init__(self,head_size,num_heads,encoder_width,**kwargs):
        super().__init__(**kwargs)
        self.ff1=Feed_forward_module(encoder_width)
        self.multihead=Multihead_module(head_size,num_heads)
        self.conv=Conv_module(encoder_width)
        self.ff2=Feed_forward_module(encoder_width)        
    def call(self,inputs):
        x=self.ff1(inputs)
#         print('conformer feedforward1')
#         print(x.shape)
        x=self.multihead(x)
#         print('conformer multihead')
#         print(x.shape)
        x=self.conv(x)
#         print('conformer convolution')
#         print(x.shape)
        x=self.ff2(x)
#         print('conformer feedforward2')
#         print(x.shape)
        return x


# In[9]:


class Encoder(keras.Model):
    def __init__(self,num_conformers,head_size,num_heads,encoder_width,**kwargs):
        super().__init__(**kwargs)
        self.conv1=keras.layers.Conv2D(1,(2,2),strides=2,input_shape=(111,2975,80,1))
        self.relu1=keras.activations.relu
        self.conv2=keras.layers.Conv2D(1,(2,2),strides=2)
        self.relu2=keras.activations.relu
        self.flat=keras.layers.Flatten()
        self.linear=keras.layers.Dense(encoder_width)
        self.dropout=keras.layers.Dropout(0.1)
        self.conformers=[Conformer_block(head_size,num_heads,encoder_width) for i in range(num_conformers)]
    def call(self,inputs,training):
        x=self.conv1(inputs)
#         print('conformer_encoder 2D convolution')
#         print(x.shape)
        x=self.relu1(x)
#         print('conformer_encoder relu activation')
#         print(x.shape)
        x=self.conv2(x)
#         print('conformer_encoder 2D convolution')
#         print(x.shape)
        x=self.relu2(x)
#         print('conformer_encoder relu activation')
#         print(x.shape)
        x=self.flat(x)
#         print('conformer_encoder flatten')
#         print(x.shape)
        x=self.linear(x)
#         print('conformer_encoder dense')
#         print(x.shape)
        x=self.dropout(x)
#         print('conformer_encoder drop out')
#         print(x.shape)
        for layer in self.conformers:
            x=layer(x)
        return x


# In[2]:


class Decoder(keras.Model):
    def __init__(self,vocab_size,decoder_width,max_len,embedding_num_dimensions,SOS=1,EOS=2,**kwargs):
        super(). __init__(**kwargs)
        self.embbeding = keras.layers.Embedding(vocab_size,embedding_num_dimensions)
        self.lstm = keras.layers.LSTM(decoder_width,return_sequences=True,return_state=True,dropout=0.1)
        self.dense = keras.layers.Dense(vocab_size,activation='softmax')
        self.SOS = SOS
        self.EOS = EOS
        self.max_len = max_len
        self.vocab_size = vocab_size
#         self.lamda = lamda
#         self.LM = LM

    def call(self,inputs,training=True):
        if training:
            x = self.embbeding(inputs[1])
            x,_,__ = self.lstm(x,initial_state=[inputs[0], inputs[0]])
            x = self.dense(x)
            return x
        else:
            current_subword = self.SOS
            h,c = inputs,inputs
            outputs = []
            for i in range(self.max_len):
                if current_subword == self.EOS:
                    break
                x = self.embbeding(current_subword)[np.newaxis,np.newaxis,:]
                x,h,c = self.lstm(x,initial_state=[h,c])
                x = self.dense(x).numpy()
                x=np.squeeze(x,axis=0)
                #print('the shape of x is ',x.shape)
#                 if i == 0:
#                     print('the shape is',np.array(self.LM.predict(np.array([self.SOS])[np.newaxis,:])).shape)
#                     y = np.array(self.LM.predict(np.array([self.SOS])[np.newaxis,:]))#np.squeeze(x,axis=0)
#                 else:
#                     print('the shape of outputs is',np.array(outputs).shape)
#                     y=np.array(self.LM.predict(np.expand_dims(np.array(outputs),axis=0)))
#                     print('the shape of y1 is ',y.shape)
# #                     y=np.squeeze(y,axis=0)
# #                     print('the shape of y2 is',y.shape)
#                 current_subword = int(np.argmax(x*self.lamda+y*(1-self.lamda)))
                current_subword= int(np.argmax(x))
                outputs.append(current_subword)
            return np.array(outputs)


# In[3]:


class Conformer_transducer(keras.Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.encoder = Encoder(num_conformers=16,head_size=16,num_heads=4,encoder_width=144)
        self.decoder = Decoder(vocab_size=977,decoder_width=144,max_len=90,embedding_num_dimensions=320,SOS=1,EOS=2)
        #self.wpm = Language_Model.WPM(vocab_source=r'C:\Users\HH\Desktop\project\newvocab2.txt')
    def call(self,inputs,training):
        if training:
            x = self.encoder(inputs[0])
            x = self.decoder([x,inputs[1]],training=True)
        else:
            x = self.encoder(inputs)
            x = self.decoder(x)
            #x = self.wpm.getWords(x)
        return x
    

