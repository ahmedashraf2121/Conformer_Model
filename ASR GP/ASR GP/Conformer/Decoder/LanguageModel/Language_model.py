#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras

class LanguageModel(keras.Model):
    def __init__(self,sentence_length,vocab_size=1003,language_model_width=4096,**kwargs):
        super().__init__(**kwargs)
        self.embedding = keras.layers.Embedding(vocab_size,100,input_length=sentence_length
                                                          ,mask_zero=True)
        self.lstm1 = keras.layers.LSTM(language_model_width,return_sequences=True)
        self.lstm2 = keras.layers.LSTM(language_model_width,return_sequences=True)
        self.lstm3 = keras.layers.LSTM(language_model_width)
        self.linear1 = keras.layers.Dense(language_model_width)
        self.linear2 = keras.layers.Dense(vocab_size,activation='softmax')
        
    def call(self,inputs):
        x = self.embedding(inputs)
        x = self.lstm1(x) 
        x = self.lstm2(x) 
        x = self.lstm3(x)
        x = self.linear1(x)
        x = self.linear2(x)
        return x

