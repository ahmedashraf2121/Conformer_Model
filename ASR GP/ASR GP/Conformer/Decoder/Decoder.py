import tensorflow as tf
from tensorflow import keras

class Decoder(keras.Model):
    def __init__(self,vocab_size,decoder_width,max_len,embedding_num_dimensions,SOS=1,EOS=2,**kwargs):
        super(). __init__(**kwargs)
        self.embbeding = keras.layers.Embedding(vocab_size,embedding_num_dimensions)
        self.lstm = keras.layers.LSTM(decoder_width,return_sequences=True,return_state=True,dropout=0.1)
        self.dense = keras.layers.Dense(vocab_size,activation='softmax')
        self.SOS = SOS
        self.EOS = EOS
        self.max_len = max_len

    def call(self,inputs,training=True):
        if training:

	    #--------the context vector coming from the encoder enters both the cell and hidden states of the LSTM-----
	    #------the input to the LSTM is the decoder input from the preprocessing function in Preprocessors.py------

            x = self.embbeding(inputs[1])
            x,_,__ = self.lstm(x,initial_state=[inputs[0], inputs[0]])
            x = self.dense(x)
            return x
        else:

	    #------during the sentence prediction the context vector enters the both the hidden and cell states of the LSTM--------
	    #------after that predicting the sentence starts subword by subword-----------------------

            current_subword = self.SOS
            h,c = inputs,inputs
            outputs = []
            for i in range(self.max_len):
                if current_subword == self.EOS:
                    break
                x = self.embbeding(current_subword)[np.newaxis,np.newaxis,:]
                x,h,c = self.lstm(x,initial_state=[h,c])
                x = self.dense(x).numpy()
                current_subword = int(np.argmax(x))
                outputs.append(current_subword)
            return np.array(outputs)
