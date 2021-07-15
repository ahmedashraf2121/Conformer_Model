#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow import keras
import tensorflow_addons as tfa
import numpy as np
import os
os.chdir(r'C:\Users\HH\Desktop\project')
import Language_Model
import Conformer_Model
import math


# In[2]:


class quick_model(keras.Model):
    def __init__(self,**kwargs):
        super().__init__(**kwargs)
        self.hidden1=Conformer_Model.Conformer_block(16,4,144)
    def call(self,inputs):
        x=self.hidden1(inputs)
        return x
model1=quick_model()
input_feed=tf.convert_to_tensor(np.random.rand(500,144))
output_feed=input_feed*5


# In[3]:


model1.compile(optimizer='adam',loss=keras.losses.MeanSquaredError(),metrics='accuracy')


# In[4]:


model1.fit(input_feed,output_feed,epochs=100)


# In[33]:


model_feed.summary()
parameters=model_feed.trainable_variables


# In[37]:


print(parameters[3])


# In[ ]:


lang_model=Language_Model.LanguageModel(sentence_length=90,vocab_size=1003,language_model_width=512)
x_lang,y_lang=textprocessor.generateBatch(r'H:\data\train clean 100\LibriSpeech\train-clean-100\19')
lang_model.compile(optimizer='rmsprop',loss=keras.losses.CategoricalCrossentropy())
lang_model.fit(x_lang,y_lang,epochs=5)


# In[10]:


# from tensorflow.compat.v1 import ConfigProto
# from tensorflow.compat.v1 import InteractiveSession

# config = ConfigProto()
# config.gpu_options.allow_growth = True
# session = InteractiveSession(config=config)
model6.fit(input2,output2,epochs=5)


# In[39]:


hello=model6.predict(np.random.rand(1,144,1))
hello=hello[0]
print(hello.shape)
print(len(hello))
new=[]
for cat in hello:
    new.append(np.argmax(cat))
print(new)


# In[10]:


parameters1=model6.trainable_variables


# In[11]:


print(parameters1[1])


# In[14]:


Full_Model=Conformer_Model.Conformer_transducer()


# In[17]:


Full_Model.compile(optimizer='adam',loss=keras.losses.CategoricalCrossentropy(),metrics='accuracy',run_eagerly=True)


# In[18]:


Full_Model.fit([x,decoder_input],target_output,epochs=5)


# In[20]:


Full_Model.predict(np.expand_dims(x[0],axis=0))


# In[ ]:


model.save_weights('my_model_weights.h5')
model.load_weights('my_model_weights.h5')


# In[40]:





# In[46]:


import h5py
os.chdir(r'C:\Users\HH\Desktop\project')
f = h5py.File("latest.h5",'r')
bias1=list(f['/conformer_encoder/conformer/conformer_encoder/conformer_encoder_block_0/conformer_encoder_block_0_ff_module_1/conformer_encoder_block_0_ff_module_1_dense_1/bias:0'])
kernel1=list(f['/conformer_encoder/conformer/conformer_encoder/conformer_encoder_block_0/conformer_encoder_block_0_ff_module_1/conformer_encoder_block_0_ff_module_1_dense_1/kernel:0'])
bias2=list(f['/conformer_encoder/conformer/conformer_encoder/conformer_encoder_block_0/conformer_encoder_block_0_ff_module_1/conformer_encoder_block_0_ff_module_1_dense_2/bias:0'])
kernel2=list(f['/conformer_encoder/conformer/conformer_encoder/conformer_encoder_block_0/conformer_encoder_block_0_ff_module_1/conformer_encoder_block_0_ff_module_1_dense_2/kernel:0'])
beta=list(f['/conformer_encoder/conformer/conformer_encoder/conformer_encoder_block_0/conformer_encoder_block_0_ff_module_1/conformer_encoder_block_0_ff_module_1_ln/beta:0'])
gamma=list(f['/conformer_encoder/conformer/conformer_encoder/conformer_encoder_block_0/conformer_encoder_block_0_ff_module_1/conformer_encoder_block_0_ff_module_1_ln/gamma:0'])
print(len(kernel1[0]))
final_min=0
final_max=0
list2=[bias1,bias2,beta,gamma]
list3=[kernel1,kernel2]
for num in list2:
    if max(num)>final_max:
        final_max=max(num)
    if min(num)<final_min:
        final_min=min(num)
for num in list3:
    for i in num:
        if max(i)>final_max:
            final_max=max(i)
        if min(i)<final_min:
            final_min=min(i)
print('final max is :',final_max,'\n')
print('final min is :',final_min)
#Full_Model.load_weights("latest.h5")


# In[35]:


import soundfile as sf
import tensorflow_io as tfio
import os
from pydub import AudioSegment
from pydub.silence import split_on_silence
audio_data1=[]
audio_data2=[]
rate=[]
rootdir = r'D:\Condomer implementation\Data\train-clean-100\train-clean-100'
max1=0
for subdir, dirs, files in os.walk(rootdir):
    for file in files:
        print(subdir)
        #print os.path.join(subdir, file)
        filepath = subdir + os.sep + file
        if filepath.endswith('.flac'):
            data2,rate2=sf.read(filepath)
            #data2 = tfio.audio.AudioIOTensor(filepath)
            #data3=AudioSegment.from_wav(filepath)
            #audio_data1.append(data2)
            temp=len(data2)
            if temp>max1:
                max_audio=data2
                max1=temp
            #audio_data2.append(data3)
            #rate.append(rate2)
#max_audio=max(audio_data1,key=len)
print(len(max_audio))


# In[37]:


import matplotlib.pyplot as plt
sample=max_audio
audio_slice = sample[100:]
print(sample)
#audio_tensor = tf.squeeze(audio_slice, axis=[-1])
tensor = tf.cast(audio_slice, tf.float32) / 32768.0
position = tfio.experimental.audio.trim(tensor, axis=0, epsilon=0.1)
start = position[0]
stop = position[1]
processed = tensor[start:stop]
fade = tfio.experimental.audio.fade(tensor, fade_in=1000, fade_out=2000, mode="logarithmic")
spectrogram = tfio.experimental.audio.spectrogram(tensor, nfft=4096, window=400, stride=1024)
mel_spectrogram = tfio.experimental.audio.melscale(spectrogram, rate=16000, mels=80, fmin=0, fmax=8000)
# type(mel_spectrogram[0][0])
# mel_spectrogram_rows=len(mel_spectrogram)
# mel_spectrogram_columns=len(mel_spectrogram[0])
# padding=np.ones((130-mel_spectrogram_rows,mel_spectrogram_columns)).astype(np.float32)
# mel_spectrogram_new=np.concatenate((mel_spectrogram,padding),axis=0)
dbscale_mel_spectrogram = tfio.experimental.audio.dbscale(mel_spectrogram, top_db=80)
print(mel_spectrogram)
plt.figure()
plt.imshow(dbscale_mel_spectrogram.numpy())
plt.figure()
plt.plot(tensor.numpy())
print(mel_spectrogram)
print(mel_spectrogram.shape)


# In[73]:


sound_file = AudioSegment.from_wav('C:/Users/HH/Downloads/19-198-0000 (online-audio-converter.com).wav')
audio_chunks = split_on_silence(sound_file, 
    # must be silent for at least half a second
    min_silence_len=500,

    # consider it silent if quieter than -16 dBFS
    silence_thresh=-16
)
print(audio_chunks)
for i, chunk in enumerate(audio_chunks):
    out_file = ".//splitAudio//chunk{0}.wav".format(i)
    print ("exporting", out_file)
    chunk.export(out_file, format="wav")


# In[2]:


import tensorflow as tf
import tensorflow_io as tfio
import soundfile as sf
import os
def Voice_Preprocessing(root,max_frames_number):
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


# In[3]:


x=Voice_Preprocessing(r'H:\data\train clean 100\LibriSpeech\train-clean-100\19',384)


# In[4]:


roots = [r'G:\data\train-clean-100',
         r'G:\data\train-clean-360',
         r'G:\data\train-other-500']
textprocessor = Language_Model.TextPreprocessing(roots,max_length=90)
target_output=textprocessor.Target_Output_Preprocessing(144,r'H:\data\train clean 100\LibriSpeech\train-clean-100\19')
decoder_input=textprocessor.Decoder_Input_Preprocessing(144,r'H:\data\train clean 100\LibriSpeech\train-clean-100\19')


# In[15]:


print(target_output.shape)


# In[31]:


np.argmax(target_output[0][0])


# In[29]:


decoder_input[0][1]


# In[6]:





# In[57]:


import h5py
fs = h5py.File('latest.h5', 'r')

fd = h5py.File('dest6.h5', 'w')
fd1=fd.create_group('conformer_encoder')
fs.copy('conformer_encoder',fd1)


# In[58]:


fd2=fd.create_group('conformer_joint')
fs.copy('conformer_joint',fd2)
fd3=fd.create_group('conformer_prediction')
fs.copy('conformer_prediction',fd3)


# In[59]:


fd4 = h5py.File('dest7.h5', 'w')
fd5 = fd4.create_group('/')


# In[60]:


f2=h5py.File('dest6.h5', 'r')
print(f2.keys())


# In[7]:


hello1=np.zeros(10)
hello2=np.zeros(10)
hello=[hello1,hello2]
print(len(hello1))


# In[16]:


f=open('newvocab1.txt','r')
file=f.read()
old_list=file.split('\n')
new_list=[]
for subword in old_list:
    if subword not in new_list:
        new_list.append(subword)
print(len(new_list))
f.close()
f=open('newvocab2.txt','a')
for subword in new_list:
    f.write(subword+'\n')
f.close()


# In[97]:


import numpy as np
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
        self.max_len=max_len

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
                current_subword = int(np.argmax(x))
                outputs.append(current_subword)
            return np.array(outputs)


# In[98]:


Decoder_model = Decoder(vocab_size=977,decoder_width=144,max_len=90,embedding_num_dimensions=320,SOS=1,EOS=2)
Decoder_model.compile(optimizer='rmsprop',loss=keras.losses.MeanSquaredError(),metrics='accuracy',run_eagerly=True)


# In[99]:


input2 = [np.random.rand(111,144),decoder_input]
output2 = target_output
Decoder_model.fit(input2,output2,epochs=1)


# In[100]:


input3 = np.random.rand(1,144)
Decoder_model.predict(input3)


# In[72]:


a=np.array([[[1,2,3],[3,4,5],[5,6,7]],[[4,5,6],[6,7,8],[9,10,11]]])
print(a)
print(np.argmax(a,axis=1))


# In[84]:


ahmed=np.array([2,5,3])[np.newaxis,np.newaxis,:]
x4=np.array([[[3]]])
print(x4)
print(int(x4))

