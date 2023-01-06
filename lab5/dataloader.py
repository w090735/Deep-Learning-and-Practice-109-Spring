from utils import *

# transformation between vocabulary and index
class DataTransformer:
    def __init__(self):
        # dict: 28 elements
        # key: "SOS", "EOS", "a"-"z"
        # value: 0, 1, 2-27
        self.char2idx = self.build_char2idx()
        # dict: 28 elements
        # key: 0, 1, 2-27
        # value: "SOS", "EOS", "a"-"z"
        self.idx2char = self.build_idx2char()
        self.tense2idx = {'sp':0,'tp':1,'pg':2,'p':3}
        self.idx2tense = {0:'sp',1:'tp',2:'pg',3:'p'}
        self.max_length = 0  # max length of the training data word(contain 'EOS')

    def build_char2idx(self):
        dictionary={'SOS':0,'EOS':1}
        # chr(ord('a') + i)
        # update: insert to dict
        dictionary.update([(chr(i+97),i+2) for i in range(0,26)])
        return dictionary

    def build_idx2char(self):
        dictionary={0:'SOS',1:'EOS'}
        # chr(ord('a') + i)
        # update: insert to dict
        dictionary.update([(i+2,chr(i+97)) for i in range(0,26)])
        return dictionary

    def string2tensor(self,string,add_eos=True):
        # index: tensor, (word length, 1)
        indices=[self.char2idx[char] for char in string]
        if add_eos:
            indices.append(self.char2idx['EOS'])
        return torch.tensor(indices,dtype=torch.long).view(-1,1)

    def tense2tensor(self,tense):
        # tense: tensor, (1)
        return torch.tensor([tense],dtype=torch.long)

    def tensor2string(self,tensor):
        # input : word index tensor, (word length, 1)
        # output: word string
        re = ""
        string_length = tensor.size(0)
        for i in range(string_length):
            char = self.idx2char[tensor[i].item()]
            if char == 'EOS':
                break
            re += char
        return re

    def get_dataset(self,path,is_train):
        # words: list, 1227x4(4908)
        # tenses: list, 1227x4(4908)
        words=[]
        tenses=[]
        with open(path,'r') as file:
            if is_train:
                for line in file:
                    # split line and word
                    words.extend(line.split('\n')[0].split(' '))
                    # 1 line has 4 tenses
                    tenses.extend(range(0,4))
            else:
                for line in file:
                    # split line and word
                    words.append(line.split('\n')[0].split(' '))
                # 10 tense transformation of test dataset
                test_tenses=[['sp','p'],['sp','pg'],['sp','tp'],['sp','tp'],['p','tp'],
                             ['sp','pg'],['p','sp'],['pg','sp'],['pg','p'],['pg','tp']]
                for test_tense in test_tenses:
                    # transform string to index
                    tenses.append([self.tense2idx[tense] for tense in test_tense])
        return words,tenses

# Dataset
class MyDataSet(data.Dataset):
    def __init__(self,path,is_train):
        # mode
        self.is_train = is_train
        self.dataTransformer = DataTransformer()
        # load data and construct list
        self.words, self.tenses = self.dataTransformer.get_dataset(os.path.join('lab5_data',path),is_train)
        # get the max length of one word
        self.max_length = self.get_max_length(self.words)
        # function: transform word to index tensor
        self.string2tensor = self.dataTransformer.string2tensor
        # function: transoform tense to tensor
        self.tense2tensor = self.dataTransformer.tense2tensor
        # function: transform word index tensor to string
        self.tensor2string = self.dataTransformer.tensor2string
        assert len(self.words) == len(self.tenses),'check if data size is correct'

    def __len__(self):
        return len(self.words)

    def __getitem__(self, idx):
        # tensor 1227x4(4908) word, tense
        if self.is_train:
            return self.string2tensor(self.words[idx],add_eos=True), self.tense2tensor(self.tenses[idx])
        # input tensor 10 word, tense
        # output tensor 10 word, tense
        else:
            return self.string2tensor(self.words[idx][0],add_eos=True), self.tense2tensor(self.tenses[idx][0]),\
                   self.string2tensor(self.words[idx][1],add_eos=True), self.tense2tensor(self.tenses[idx][1])

    def get_max_length(self,words):
        max_length=0
        for word in words:
            max_length=max(max_length,len(word))
        return max_length
    