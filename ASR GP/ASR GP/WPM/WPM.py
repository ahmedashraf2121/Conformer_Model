from VocabTree.VocabTree import VocabTree


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
        Word_list=[]
        for ID in ID_list :
            Word_list += self.vocab_tree.findWord(ID)
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
