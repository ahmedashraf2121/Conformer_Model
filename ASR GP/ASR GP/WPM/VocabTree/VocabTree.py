from Node.node import node

class VocabTree():
    def __init__(self,vocab):
        self.vocab = vocab
        self.root = node('*')
        self.build()

    #-------the function used to build the vocab tree from a vocab already existing---------

    def build(self):
        for i in range(len(self.vocab)):
            self.root.insertWord(self.vocab[i],i)

    #-------the function used to find the ID for certain word from the vocab tree-------

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

    #-------the function used to obtain a word corresponding to a specific ID in the vocab tree---------

    def findWord(self,ID):
        return self.vocab[ID]

    def printTree(self):
        self.root.printNode()
