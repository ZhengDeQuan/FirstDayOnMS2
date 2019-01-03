from tqdm import tqdm
import gensim.models as g
import pickle
from typing import List
import logging
import json
import os
from collections import Counter
from jieba import analyse
import gensim.models as g
gensim_weight_save_path = "../SongPoem2Vec/songpoem2vec_custom_wordvec_song.bin"

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MatchPoemSong:
    def __init__(self , poem_file:str , song_file:str , keywords:List[str] ,
                 base_dir_for_save = r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem_Song',
                 save_dir_for_songVec = "songVec.pkl",
                 save_dir_for_poemVec = "poemVec.pkl"):
        self.poem_file = poem_file
        self.song_file = song_file
        self.base_dir_for_save = base_dir_for_save
        self.save_dir_for_songVec = save_dir_for_songVec
        self.save_dir_for_poemVec = save_dir_for_poemVec
        self.keywords = keywords
        self.poems = []
        self.sub_poems = []
        self.songs = []
        self.sub_songs = []
        self.songVecs = []
        self.poemVecs = []

    def filterPoem(self,para_dict):
        '''
        :param para_dict: 每个小散文的dict
        :return: bool True is this para belongs to the topic defined in self.keywords False otherwise
        '''
        festival_count = Counter(para_dict['festival'])
        season_count = Counter(para_dict['season'])
        to_be_test = Counter(dict(para_dict['key_words']))

        res1 = to_be_test & self.keywords_count
        res2 = festival_count & self.keywords_count
        res3 = season_count & self.keywords_count
        res = res1 | res2 | res3
        return sum(res.values())

    def getPoem(self):
        with open(self.poem_file,"r",encoding='utf8') as fout:
            self.poems = json.load(fout)

        self.keywords_count = Counter(self.keywords)
        self.sub_poems = []
        for poem_dict in self.poems:
            poem_id = poem_dict['poem_id']
            for para_id , para_dict in enumerate(poem_dict['paras']):
                score =  self.filterPoem(para_dict)
                if score < 0.001:
                    continue
                content = []
                title = para_dict['fencied_para_title']
                title = ' '.join(title).strip()
                if title:
                    content.append(title)
                for para_c in para_dict['fencied_para_content']:
                    content.append(' '.join(para_c))
                temp = {'poem_id':poem_id,"para_id":para_id,'match_score_with_topic':score,"content":'\n'.join(content)}
                self.sub_poems.append(temp)

    def getKeyWordFromText(self,text):
        res = analyse.textrank(text, topK=10, withWeight=True, allowPOS=('ns', 'n', 'vn'))
        # res = analyse.textrank(text, topK=10, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
        return res

    def filterSong(self, key_words, text, threshold=0.99):
        res = self.getKeyWordFromText(text)
        res = dict(res)
        res = Counter(res)
        key_words = Counter(key_words)
        a = res & key_words
        score = sum(a.values())
        # score = score / len(list(jieba.cut(text)))
        if score > threshold:
            return (True, score)
        return (False, score)

    def getSong(self):
        self.songs = pickle.load(open(self.song_file,"rb"))
        self.keywords_count = Counter(self.keywords)
        self.sub_songs = []
        for oneSong in tqdm(self.songs):
            music_name = oneSong['music_name']['ori_name']
            lyric = oneSong['lyric']['ori_lyric']
            Flag, score = self.filterSong(self.keywords_count, music_name + "\t" + lyric, threshold=0.99)
            if Flag:
                print("score = ", score)
                oneSong["match_score_with_topic"]=score
                self.sub_songs.append(oneSong)

    def Prepro(self, top_p=100, top_s=100):
        self.TextPoem = []
        self.TextSong = []
        for i , poem in enumerate(self.sub_poems[:top_p]):
            self.TextPoem.append(poem["content"])
        for i , song in enumerate(self.sub_songs[:top_s]):
            self.TextSong.append(song['lyric']['wb_lyric'])

    def Text2Vec(self):
        pass

    def MatchPoemSong(self):
        '''
        1。 先对歌曲和散文排序 ，以match_score_with_topic为关键字
        2. 再取前100-200的歌曲和散文
        3. 从歌曲对象--》歌曲歌词，散文对象--》散文文字
        4.得到从句子到向量的转换
        :return:
        '''
        self.sub_songs.sort(key = lambda song:song['match_score_with_topic'],reverse=True)
        self.sub_poems.sort(key = lambda poem:poem['match_score_with_topic'],reverse=True)
        # for i , song in enumerate( self.sub_songs ):
        #     print(song['match_score_with_topic'])
        #     if i == 10:
        #         break
        #
        # for i , poem in enumerate( self.sub_poems):
        #     print(poem['match_score_with_topic'])
        #     if i == 10:
        #         break
        #
        # print(len(self.sub_poems))
        # print(len(self.sub_songs))
        self.Prepro(top_p=100,top_s=100)
        self.Text2Vec()



    def save(self):
        pass

    def load(self):
        pass

    def forward(self):
        # self.getPoem()
        # pickle.dump(self.sub_poems,open(os.path.join(self.base_dir_for_save,"sub_poems.pkl"),"wb"))
        # self.getSong()
        # pickle.dump(self.sub_songs,open(os.path.join(self.base_dir_for_save,"sub_songs.pkl"),'wb'))
        self.sub_songs = pickle.load(open(os.path.join(self.base_dir_for_save,"sub_songs.pkl"),"rb"))
        self.sub_poems = pickle.load(open(os.path.join(self.base_dir_for_save,"sub_poems.pkl"),"rb"))
        print(len(self.sub_poems))  # 211
        print(len(self.sub_songs))  # 201
        self.MatchPoemSong()

if __name__ == "__main__":
    poem_song_matcher = MatchPoemSong(poem_file=os.path.join('E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem',"processed_poem.json"),
                                      song_file=os.path.join('E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Song',"song4.pkl",),
                                      keywords=['夜晚','深夜', '寂静', '安眠', '星空', '平静', '喧嚣','静','夜色','月亮',"失眠"])
    poem_song_matcher.forward()
    # a = [(2.5,1),(3.4,11),(7.1,12)]
    # b = [(2.5,2),(3.3,22),(8.9,222)]
    # c = [(2.5,57),(3.3,88),(9.9,89)]
    # Temp1 = {'poem_id':1,
    #         'match_score':a}
    # Temp2 = {
    #     "poem_id":2,
    #     'match_score':b
    # }
    # Temp3 = {
    #             "poem_id":3,
    #         'match_score':c
    # }
    # Li = [Temp1, Temp2,Temp3]
    # aa,ba = zip(*a)
    # print(aa)
    # print(ba)
    # Li_so = sorted(Li,key = lambda Temp:list(zip(*Temp['match_score']))[0])
    # print(Li_so)