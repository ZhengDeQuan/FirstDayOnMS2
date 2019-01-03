'''
PoemToText
将preprocessed_poem.json中的数据变成每个para一行。同时制作一个两个map表，(poem_id,para_id)<-->(doc_id)
doc_id 是在doc_vec表中的行方向的index
'''
import os
import json
import pickle
import numpy as np
import torch
from torch import nn
import random
import collections
from collections import deque
from itertools import compress

class PoemToText:
    def __init__(self, base_dir = "E:\\PycharmProjects\\FirstDayOnMS2\\Data" ,file_name = "Poem/processed_poem.json" , save_dir = "Poem_Song"):
        super(PoemToText, self).__init__()
        self.base_dir = base_dir
        self.file_name = file_name
        self.save_dir = save_dir

    def load_poem(self):
        if not os.path.join(os.path.join(self.base_dir,self.file_name)):
            raise ValueError(" the load dir doesn't exist")
        with open(os.path.join(self.base_dir,self.file_name),"r",encoding="utf8") as fin:
            self.poems = json.load(fin)
        self.sub_poem = []
        self.doc_id = 0
        self.doc_id2tuple = dict() #tuple means (poem_id , para_id)
        self.tuple2doc_id = dict()
        self.doc_id2doc_len = dict()
        self.max_para_length = 0
        for poem_dict in self.poems:
            poem_id = poem_dict['poem_id'] # start from 1
            for para_id , para_dict in enumerate( poem_dict['paras']):
                temp = [''.join(para) for para in para_dict['fencied_para_content']]
                content = ''.join(temp)
                self.max_para_length = max(self.max_para_length,len(content))
                self.sub_poem.append(content)
                self.doc_id2tuple[self.doc_id] = (poem_id,para_id)
                self.doc_id2doc_len[self.doc_id] = len(content)
                self.tuple2doc_id[(poem_id,para_id)] = self.doc_id
                self.doc_id += 1
        self.doc_num = self.doc_id
        print("num_doc = ",self.doc_num)
        print("max_para_length = ",self.max_para_length)
        print("self.doc_id2doc_len = ",self.doc_id2doc_len)

    def save(self):
        with open(os.path.join(self.base_dir,self.save_dir,"doc_for_bert.txt"),"w",encoding="utf8") as fout:
            for doc_id , content in enumerate( self.sub_poem ):
                #fout.write(str(self.doc_id2tuple[doc_id]) + " "+  content+"\n") # 一个para一行
                fout.write(content + "\n")  # 一个para一行
        with open(os.path.join(self.base_dir,self.save_dir,"doc_id2tuple.pkl"),"wb") as fout:
            pickle.dump(self.doc_id2tuple,fout)
        with open(os.path.join(self.base_dir,self.save_dir,"tuple2doc_id.pkl"),"wb") as fout:
            pickle.dump(self.tuple2doc_id,fout)


    def forward(self):
        self.load_poem()
        self.save()


class Doc2VecWithBert:
    def __init__(self ,base_dir , file_name , save_dir , bert_vec_file,
                 num_docs , vec_dim , seed = 2018, noisy_word_pool_size = 100,
                 batch_size = 20 , window_size = 4):
        super(Doc2VecWithBert,self).__init__()
        self.base_dir = base_dir
        self.file_name = file_name
        self.save_dir = save_dir
        self.bert_vec_file = bert_vec_file
        self._DocVec = nn.Parameter(
            torch.randn(num_docs, vec_dim), requires_grad=True)
        random.seed(seed)
        np.random.seed(seed)
        self.noisy_word_pool = noisy_word_pool_size
        self.batch_size = batch_size
        self.window_size = window_size

    def get_paragraph_vector(self, index):
        return self._DocVec[index, :].data.tolist()

    def getNoisyWordPool(self):
        self.noisy_word_pool = deque(maxlen=self.noisy_word_pool_size)
        choose = np.random.randint(self.num_doc // 2, self.num_doc, self.noisy_word_pool).tolist()
        with open(os.path.join(self.base_dir, self.save_dir, self.bert_vec_file), "r") as fin:
            for i, line in enumerate(fin):
                if i in choose:
                    line_dict = json.loads(line)
                    assert i == line_dict['index']
                    features = line[
                        'features']  # list[ele1,~elen] 长度与句子长度+2一样，ele = dict(['token','layers']) 'layers' = [l1,l2,l3,l4] 长度为4，因为输入中，我们用了四个层 ,\
                    # print(len(features))  # 所限定的最长的长度512
                    # l1 = dict_keys(['index', 'values']) index指示的是第几层，-1，-2，-3，-4
                    # 'values' = list [768维度] 就是这个词在这个上下文环境下的在这一层的向量表示了。
                    temp = []
                    for j, item in enumerate(features):
                        temp.append((item['token'], item['layers'][0]['values']))  # 0是倒数第一层的输出
                    self.noisy_word_pool.append(temp)

    def getWordandDoc(self):
        self.words = [] #ele is tuple (doc_id,word_indx)
        self.docs = [] #ele is int , len(docs) == len(words)
        for poem_id , poem in enumerate( self.sub_poems ):
            self.docs.extend([poem_id] * len(poem))
            for word_id , word in poem:
                self.words.append((poem_id,word_id))

    def load_corpus(self):
        with open(os.path.join(self.base_dir,self.file_name),"r",encoding="utf8") as fin:
            self.sub_poems = fin.readlines()
        with open(os.path.join(self.base_dir,self.save_dir,"doc_id2tuple.pkl"),"rb") as fin:
            self.doc_id2tuple = pickle.load(fin)
        with open(os.path.join(self.base_dir,self.save_dir,"tuple2doc_id.pkl"),"rb") as fin:
            self.tuple2doc_ic = pickle.load(fin)
        self.num_doc = len(self.doc_id2tuple)
        self.getNoisyWordPool()
        self.getWordandDoc()

    def generate_batch_pvdm(self, batch_size, window_size):
        '''
        Batch generator for PV-DM (Distributed Memory Model of Paragraph Vectors).
        batch should be a shape of (batch_size, window_size+1)

        Parameters
        ----------
        batch_size: number of words in each mini-batch
        window_size: number of leading words on before the target word direction
        '''
        assert batch_size % window_size == 0
        batch = np.ndarray(shape=(batch_size, window_size + 1), dtype=np.int32)
        labels = [() for i in range(batch_size)]
        span = window_size + 1
        buffer = collections.deque(maxlen=span)  # used for collecting word_ids[data_index] in the sliding window
        buffer_doc = collections.deque(maxlen=span)  # collecting id of documents in the sliding window
        # collect the first window of words
        for _ in range(span):
            buffer.append(self.words[self.data_index])
            buffer_doc.append(self.docs[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.words)

        mask = [1] * span
        mask[-1] = 0
        i = 0
        desired_min_doc_id , desired_max_doc_id = None, None
        while i < batch_size:
            if len(set(buffer_doc)) == 1:
                doc_id = buffer_doc[-1]
                if desired_min_doc_id :
                    desired_max_doc_id = doc_id
                    desired_min_doc_id = doc_id
                elif doc_id < desired_min_doc_id:
                    desired_min_doc_id = doc_id
                elif doc_id > desired_max_doc_id:
                    desired_max_doc_id = doc_id

                # all leading words and the doc_id
                batch[i, :] = list(compress(buffer, mask)) + [doc_id]
                labels[i] = buffer[-1]  # the last word at end of the sliding window
                i += 1
            # print buffer
            # print list(compress(buffer, mask))
            # move the sliding window
            buffer.append(self.words[self.data_index])
            buffer_doc.append(self.docs[self.data_index])
            self.data_index = (self.data_index + 1) % len(self.words)

        return batch, labels , desired_min_doc_id , desired_max_doc_id

    def generate_bert_vec(self):
        with open(os.path.join(self.base_dir, self.save_dir, self.bert_vec_file), "r", encoding="utf8") as fin:
            for i, line in enumerate(fin):
                line = json.loads(line)
                features = line['features']
                temp = []
                for j, item in enumerate(features):
                    temp.append((item['token'],item['layers'][3]['values']))
                yield (i, temp)

    def Train(self):
        self.data_index = 0
        Batch = self.generate_bert_vec()
        while True:
            batch, labels, desired_min_doc_id, desired_max_doc_id = self.generate_batch_pvdm(self.batch_size , self.window_size)
            cur_doc_id , doc_word_vec = next(Batch)



    def save(self):
        pass



    def forward(self):
        self.load_corpus()
        self.Train()

def getFromFile():
    base_dir = "E:\\PycharmProjects\\FirstDayOnMS2\\Data"
    save_dir = "Poem_Song"
    with open(os.path.join(base_dir, save_dir, "doc_for_bert.txt"), "r", encoding="utf8") as fin:
        for i , line in enumerate(fin):
            yield(i , line)

if __name__ == "__main__":
    poemToText = PoemToText()
    poemToText.forward()
    # Batch = getFromFile()
    # now_min_doc_id = 0
    # now_max_doc_id = 0
    # desired_min_doc_id = 0
    # desired_max_doc_id = 1
    # doc_pool = []
    # for kk in range(5):
    #     if now_min_doc_id < desired_min_doc_id:
    #         now_min_doc_id = desired_min_doc_id
    #     while now_max_doc_id < desired_max_doc_id:
    #         cur_doc_id , doc = next(Batch)
    #         now_max_doc_id = cur_doc_id
    #         print(now_max_doc_id)
    #         print(doc)




