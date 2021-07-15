import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import os
import numpy as np
import tensorflow_io as tfio
import soundfile as sf

class TextPreprocessing():
    def __init__(self,roots,file_suffix='.trans.txt',max_length=None):
        self.text_encoder = WPM(vocab_source=r'C:\Users\HH\Desktop\project\newvocab2.txt')
        self.file_suffix = file_suffix
        self.max_length = max_length
        self.roots = roots
        if max_length == None:
            self.findMaxLength()
            
   #----------a function that finds the number of subwords in the longest sentence---------
     
    def findMaxLength(self):
        for root in self.roots:
            for subdir,dirs,files in os.walk(root):
                for file in files:
                    if file.endswith(self.file_suffix):
                        text = self.getText(subdir+os.sep+file)
                        for line in text:
                            ids = self.text_encoder.getIds(line)
                            line_length = len(line)
                            if line_length > self.max_length:
                                self.max_length = line_length

#-------a function that reads the sentences in a file and puts an SOS and EOS at the end-----------
        
    def getText(self,dir):
        File = open(dir,'r')
        text = File.read()
        text = text.split('\n')
        text_samples = []
        del text[-1]
        for sent in text:
            text_samples.append(sent.split(' '))
        for sent in text_samples:
            sent[0] = '[SOS]'
            sent.append('[EOS]')
        return text_samples

#-----------a function that generates the training batch for the language model-------------
        
    def generateBatch(self,root):
        IDS = []
        x,y = [],[]
        for subdir,dirs,files in os.walk(root):
            for file in files:
                if file.endswith(self.file_suffix):
                    text = self.getText(subdir+os.sep+file)
                    for sent in text:
                        IDS.append(self.text_encoder.getIds(sent))
                    for ID_line in IDS:
                        for i in range(len(ID_line)-1):
                            x.append(ID_line[:i+1])
                            y.append(ID_line[i+1])
        x = np.array(pad_sequences(x,maxlen=self.max_length,padding='post'))
        y = np.array(y)
        y = keras.utils.to_categorical(y,num_classes=len(self.text_encoder.vocab))
        return x,y

#-----------a function that generates the target output of the decoder of the main model----------
    
    def Target_Output_Preprocessing(self,max_length,root):
        output=[]
        for subdir,dirs,files in os.walk(root):
                    for file in files:
                        if file.endswith(self.file_suffix):
                            text = self.getText(subdir+os.sep+file)
                            for line in text:
                                ids = self.text_encoder.getIds(line)
                                output.append(ids)
        for line in output:
            line.remove(1)
        padded_target_output=pad_sequences(output,maxlen=max_length,padding='post')
        padded_target_output=keras.utils.to_categorical(padded_target_output,num_classes=977)
        padded_target_output=np.array(padded_target_output)
        padded_target_output=tf.convert_to_tensor(padded_target_output)
        return padded_target_output

#-------a function that generates the input to the LSTM of the decoder----------
#-------the input to the LSTM of the decoder is the same as the target output but ahead of it one time step due to the SOS in the beginning--------
    
    def Decoder_Input_Preprocessing(self,max_length,root):
        output=[]
        for subdir,dirs,files in os.walk(root):
                    for file in files:
                        if file.endswith(self.file_suffix):
                            text = self.getText(subdir+os.sep+file)
                            for line in text:
                                ids = self.text_encoder.getIds(line)
                                output.append(ids)
        padded_decoder_input=pad_sequences(output,maxlen=max_length,padding='post')
        padded_decoder_input=np.array(padded_decoder_input)
        padded_decoder_input=tf.convert_to_tensor(padded_decoder_input)
        return padded_decoder_input

#------preparing the input to the model starting from a certain root and with certain maximum number of frames for padding-------

    def Voice_Preprocessing(self,root,max_frames_number):
        spec_data=[]
        for subdir,dirs,files in os.walk(root):
                for file in files:
            		filepath=subdir+os.sep+file
            		if filepath.endswith('.flac'):
                		data,rate = sf.read(filepath)
				data = tf.convert_to_tensor(data) 
                		tensor = tf.cast(data, tf.float32) / 32768.0
                		position = tfio.experimental.audio.trim(tensor, axis=0, epsilon=0.1)
                		processed = tensor[position[0]:position[1]]
                		fade = tfio.experimental.audio.fade(processed, fade_in=1000, fade_out=2000, mode="logarithmic")
                		spectrogram = tfio.experimental.audio.spectrogram(fade, nfft=4096, window=400, stride=160)
                		mel_spectrogram = tfio.experimental.audio.melscale(spectrogram, rate=16000, mels=80, fmin=0, fmax=8000)
                		mel_spectrogram_rows=len(mel_spectrogram)
                		mel_spectrogram_columns=len(mel_spectrogram[0])
                		padding=np.ones((max_frames_number-mel_spectrogram_rows,mel_spectrogram_columns)).astype(np.float32)
                		mel_spectrogram_new=np.concatenate((mel_spectrogram,padding),axis=0)
                		dbscale_mel_spectrogram = tfio.experimental.audio.dbscale(mel_spectrogram_new, top_db=80)
                		spec_data.append(tf.expand_dims(dbscale_mel_spectrogram,axis=-1))
    	        spec_data=np.array(spec_data)
        return spec_data