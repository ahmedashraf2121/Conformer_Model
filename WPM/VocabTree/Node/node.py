#-------The vocab tree consists of nodes and each node has child nodes, each node carries a character------

class node():
    def __init__(self,key,value=None):
        self.key = key
        self.value = value
        self.child = []

    #-------a function to add a child node to the current node------- 

    def addChild(self,key,value=None):
        self.child.append(node(key,value))

    #------This function inserts a word into the tree character by character-------
    #------Where the last character in each word has the numerical value of this word----
    #------Which means that the first generation of nodes represents the main alphabetical characters---

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

    #--------this function finds the value of the word you are looking for and this value will be carried by the last character in the word---
    #-------and if the word doesn't exist then the value or ID will be None--------------

    def findMatch(self,string):
        length = len(string)
        ID = None
        Lleft = length
        for child in self.child:
            if child.key == string[0] and length == 1:#you have reached the last character
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

    #-------print all the characters and values associated with it in the tree----------- 

    def printNode(self):
        print(self.key,' ',self.value,end = ':\n')
        for child in self.child:
            print(child.key,' ',child.value,end = '  ')
        print('\n-----------------------***-----------------------')
        for child in self.child:
            child.printNode()
