#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow import keras
import os
import numpy as np


# In[2]:


class node():
    def __init__(self,key,value=None):
        self.key = key
        self.value = value
        self.child = []
        
    def addChild(self,key,value=None):
        self.child.append(node(key,value))
        
    def insertWord(self,str,value=None):
        length = len(str)
        if length == 1:
            childFound = False
            for child in self.child:
                if child.key == str:
                    child.value = value
                    childFound = True
            if not childFound:
                self.addChild(str,value)
        else:
            isFound = False
            for child in self.child:
                if str[0] == child.key:
                    isFound = True
                    child.insertWord(str[1:],value)
            if not isFound:
                self.addChild(str[0])
                self.child[-1].insertWord(str[1:],value)
        
    def findMatch(self,string):
        length = len(string)
        ID = None
        Lleft = length
        for child in self.child:
            if child.key == string[0] and length == 1:
                ID = child.value
                Lleft = length-1
                break
            elif child.key == string[0] and length > 1:
                ID,Lleft = child.findMatch(string[1:])
                break
        if ID == None:
            ID = self.value
            Lleft = length
        return ID,Lleft
        
    def printNode(self):
        print(self.key,' ',self.value,end = ':\n')
        for child in self.child:
            print(child.key,' ',child.value,end = '  ')
        print('\n-----------------------***-----------------------')
        for child in self.child:
            child.printNode()


# In[3]:


class VocabTree():
    def __init__(self,vocab):
        self.vocab = vocab
        self.root = node('*')
        self.build()
        
    def build(self):
        for i in range(len(self.vocab)):
            self.root.insertWord(self.vocab[i],i)
        
    def findID(self,str):
        ID = []
        isStart = True
        length = len(str)
        while(length > 0):
            if isStart:
                matchId,leftLength = self.root.findMatch(str[-length:])
            else:
                matchId,leftLength = self.root.findMatch('_'+str[-length:])
            ID.append(matchId)
            length = leftLength
            isStart = False
        return ID
    
    def findWord(self,ID):
        return self.vocab[ID]
    
    def printTree(self):
        self.root.printNode()


# In[4]:


class WPM():
    def __init__(self,root_dir=None,vocab_source=None,vocab_max_size=1000,file_suffix=r".trans.txt"
                     ,auto_find_vocab=True,score_threshold=0,fresh_model_mode=False):
        if fresh_model_mode:
            self.calcScores_ptr = 0
            self.vocab_max_size = vocab_max_size
            self.vocab_current_size = 0
            self.Root_dir = root_dir
            self.file_suffix = file_suffix
            self.score_threshold = score_threshold
            self.vocab = self.BuildStartingVocab()
            self.candidateWordPieces = self.BuildStartCandidateWPs()
            self.getStartScores()
            if auto_find_vocab:
                self.GetFullVocab()
        else:
            self.vocab = self.load(vocab_source)
            self.vocab_tree = VocabTree(self.vocab)
        
    def BuildStartingVocab(self):
        vocab = string.ascii_uppercase + "'"
        vocab = list(vocab)
        vocab_extended = []
        for subword in vocab:
            vocab_extended.append("_"+subword )
        vocab += vocab_extended
        self.vocab_current_size=len(vocab)
        return vocab
    
    def getIndex(self,char):
        if char == "'":
            char = chr(ord('Z') + 1)
        return ord(char )- ord('A')    
    
    def updateAll(self,update_sample):
        for word in update_sample:
            middle_word = 0
            for i in range(len(word)-1):
                index = self.getIndex(word[i])*27 + self.getIndex(word[i+1]) + middle_word*27*27
                self.candidateWordPieces[index][1] += 1
                middle_word = 1
    
    def getStartScores(self):
        for current_dir,dirs,files in os.walk(self.Root_dir):
            for file in files:
                if file.endswith(self.file_suffix):
                    data_sample = self.extractWords(current_dir+os.sep+file)
                    self.updateAll(data_sample)
        print('Start score is calculated.')
        for wordpiece in self.candidateWordPieces:
            if wordpiece[1] < self.score_threshold:
                del wordpiece
        self.calcScores_ptr = len(self.candidateWordPieces)
        
    
    def BuildStartCandidateWPs(self):
        outputs = []
        for wordpiece1 in self.vocab:
            for wordpiece2 in self.vocab:
                if not is_start(wordpiece2):
                    outputs.append([wordpiece1+wordpiece2[1:],0])
        return outputs

    
    def UpdateCandidateWPs(self):
        newWP = self.vocab[-1]
        self.calcScores_ptr = len(self.candidateWordPieces)
        for  subword in self.vocab:
            if not is_start(subword):
                self.candidateWordPieces.append([newWP+subword[1:],0])
                
    def extractWords(self,dir):
        file = open(dir)
        txt = file.read()
        txt = txt.split('\n')
        text = []
        for sent in txt:
            txt1 = sent.split(' ')
            text += [word for word in txt1 if isword(word)]
        return text
    
    def numberOfWordsInFile(str,data_sample):
        count = 0
        length = len(str)
        if is_start(str):
            for word in data_sample:
                    if word[:length] == str:
                        count += 1            
        else:
            str = str[1:]
            for word in data_sample:
                if word.find(str,1) > -1:
                    count += 1
        return count
    
    def countOccurences(self,str):
        numberOfOccurences = 0
        for current_dir,dirs,files in os.walk(self.Root_dir):
            for file in files:
                if file.endswith(self.file_suffix):
                    data_sample = extractWords(current_dir+os.sep+file)
                    numberOfOccurences += numberOfWordsInFile(str,data_sample)
        return numberOfOccurences
    
    def eliminateCondition(self,wordpiece):
        if is_start(wordpiece):
            word = '_' + wordpiece[1:]
        else:
            word = '_' + wordpiece[2:]
        for wrd in self.vocab:
            if wrd == word:
                return False
        for wrd in self.candidateWordPieces:
            if wrd == word and wrd[1] > self.score_threshold:
                return False
        return True
        
    
    def calcScores(self):
        for wordpiece in self.candidateWordPieces[self.calcScores_ptr:]:
            if self.eliminateCondition(wordpiece[0]):
                del wordpiece
            else:
                wordpiece[1] = self.countOccurences(wordpiece[0])
                print(wordpiece,' ')
                if wordpiece[1] == 0:
                    del wordpiece
    
    def GetFullVocab(self):
        while self.vocab_current_size < self.vocab_max_size:
            if not self.candidateWordPieces:
                break
            self.candidateWordPieces.sort(reverse=True,key=lambda x : x[1])
            self.vocab.append(self.candidateWordPieces[0][0])
            self.vocab_current_size += 1
            print('The newly added word ',self.candidateWordPieces[0],' ',self.vocab_current_size,'\n')
            del self.candidateWordPieces[0]
            neededWPs = self.vocab_max_size - self.vocab_current_size
            if len(self.candidateWordPieces)>neededWPs:
                del self.candidateWordPieces[neededWPs:]
            self.UpdateCandidateWPs()
            self.calcScores()
        self.vocab_tree = VocabTree(self.vocab)
            
    def printVocab(self):
        print(self.vocab)
        
    def getIds(self,vocab):
        Id_list = []
        for word in vocab :
            Id_list += (self.vocab_tree.findID(word))
        return Id_list
    
    def getWords(self,ID_list):
        subWord_list=[]
        Word_list=[]
        for ID in ID_list :
            subWord_list.append(self.vocab_tree.findWord(ID))
        
        return Word_list
    
    def saveas(self,file):
        File = open(file,'a')
        for word in self.vocab:
            File.write(word+'\n')
        File.close()
    
    def load(self,file):
        File = open(file,'r')
        txt = File.read()
        vocab = txt.split('\n')
        File.close()
        return vocab


# In[5]:


class LanguageModel(keras.Model):
    def __init__(self,sentence_length,vocab_size=1003,language_model_width=512,**kwargs):
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


# In[1]:


class TextPreprocessing():
    def __init__(self,roots,file_suffix='.trans.txt',max_length=None):
        self.text_encoder = WPM(vocab_source=r'C:\Users\HH\Desktop\project\newvocab2.txt')
        self.file_suffix = file_suffix
        self.max_length = max_length
        self.roots = roots
        if max_length == None:
            self.findMaxLength()
            
        
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
        


# In[ ]:




