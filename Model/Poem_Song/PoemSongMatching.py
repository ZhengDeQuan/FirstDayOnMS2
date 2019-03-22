from tqdm import tqdm
import gensim.models as g
import pickle
from typing import List
import logging
import json
import os
import time
import numpy as np
from collections import Counter
from jieba import analyse
import openpyxl
import gensim.models as g
from Poem_Song.config import opt
import requests
import yaml

gensim_weight_path = 'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem_Song\\gensim_weight\\songpoem2vec_custom_wordvec_song.bin'
bert_weight_path_base = 'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem_Song\\chinese_L-12_H-768_A-12'
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

class MatchPoemSong:
    def __init__(self ,opt, poem_file:str , song_file:str , keywords:List[str] ,
                 base_dir_for_save = r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem_Song',
                 save_dir_for_songVec = "songVec.pkl",
                 save_dir_for_poemVec = "poemVec.pkl",
                 idf_path = None):
        self.poem_file = poem_file
        self.song_file = song_file

        self.opt = opt

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
        self.model = g.Doc2Vec.load(gensim_weight_path)
        self.idf_path = idf_path
        if idf_path is not None:
            analyse.set_idf_path(idf_path)

    def filterPoem(self,para_dict):
        '''
        :param para_dict: 每个小散文的dict
        :return: bool True is this para belongs to the topic defined in self.keywords False otherwise
        '''
        festival_count = Counter(para_dict['festival'])
        season_count = Counter(para_dict['season'])
        to_be_test = Counter(dict(para_dict['idf_key_words']))

        res1 = to_be_test & self.keywords_count
        res2 = festival_count & self.keywords_count
        res3 = season_count & self.keywords_count
        res = res1 | res2 | res3
        # print("score = ",sum(res.values()))
        return sum(res.values())

    def getPoem(self):
        with open(self.poem_file,"r",encoding='utf8') as fin:
            self.poems = json.load(fin)

        self.keywords_count = Counter(self.keywords)
        self.sub_poems = []
        for poem_dict in self.poems:

            poem_id = poem_dict['poem_id']
            for para_id , para_dict in enumerate(poem_dict['paras']):
                score =  self.filterPoem(para_dict)
                if poem_dict["poem_class"] in self.keywords:
                    score += 10 + self.opt['poem_threshold']
                if score < self.opt['poem_threshold']: #0.001
                    continue
                content = []
                title = para_dict['fencied_para_title']
                title = ' '.join(title).strip()
                for para_c in para_dict['fencied_para_content']:
                    content.append(' '.join(para_c))

                temp = {'poem_id':poem_id,"para_id":para_id,'match_score_with_topic':score,"content":'\n'.join(content)}
                temp = {'poem_id':poem_id,"para_id":para_id,'match_score_with_topic':score,"content":'\n'.join(content),"url":poem_dict['url']}
                '''
                为了输出数据，查看，而有
                '''
                self.sub_poems.append(temp)
        self.sub_poems.sort(key=lambda poem: poem['match_score_with_topic'], reverse=True)

    def getKeyWordFromText(self,text):
        res = analyse.extract_tags(text, topK=10, withWeight=True, allowPOS=('ns', 'n', 'vn', 'v'))
        return res

    def filterSong(self, key_words, text, threshold=None):
        if threshold is None:
            threshold = self.opt['song_threshold']
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
            Flag, score = self.filterSong(self.keywords_count, music_name + "\t" + lyric, threshold=self.opt['song_threshold'])
            if Flag:
                # print("score = ", score)
                oneSong["match_score_with_topic"]=score
                self.sub_songs.append(oneSong)
        self.sub_songs.sort(key=lambda song: song['match_score_with_topic'], reverse=True)
        pickle.dump(self.sub_songs,open(os.path.join(self.base_dir_for_save,"sub_songs"+str(len(self.sub_songs))+".pkl"),'wb'))
        json.dump(self.sub_songs,open(os.path.join(self.base_dir_for_save,"sub_songs"+str(len(self.sub_songs))+".json"),'w',encoding="utf-8"), ensure_ascii=False)
        print("len_new_sub_songs = ",len(self.sub_songs))

    def Prepro(self, top_p=100000000, top_s=1000000000):
        self.TextPoem = []
        self.TextSong = []
        for i , poem in enumerate(self.sub_poems[:top_p]):
            self.TextPoem.append(poem["content"])
        for i , song in enumerate(self.sub_songs[:top_s]):
            self.TextSong.append(song['lyric']['wb_lyric'])

    def Text2Vec(self,Texts):
        Vecs = []
        for text in Texts:
            print("text = ", text)
            vec = self.model.infer_vector(text.split())
            print("vec = ", vec)
            Vecs.append(vec)
        return Vecs

    def makeTextForBert(self, Texts , out_file):
        '''
        betr的最长的长度是512，所以如果长度超过这个就需要进行裁剪，将一个长度较长的句子，裁剪为多个长度<=500的句子，
        得到句子向量后，去除首尾的特殊符号的embedding，然后拼接回来。
        为了之后拼接的顺利进行，需要保存一个记录表
        dict_o2s={origin_index,index_after_splited}
        dict_s2o={index_after_splited,origin_index}
        '''
        dict_o2s = {}
        dict_s2o = {}
        index_after_splited = 0
        for origin_index , text in enumerate( Texts ):
            text = text.split('\n')

    def getBertVec(self):
        '''
        python extract_features.py --input_file=./doc_for_bert.txt --output_file=./bert_vec.json --vocab_file=../chinese_L-12_H-768_A-12/vocab.txt --bert_config_file=../chinese_L-12_H-768_A-12/bert_config.json --init_checkpoint=../chinese_L-12_H-768_A-12/bert_model.ckpt --layers=-1 --max_seq_length=510 --batch_size=1
        :return:
        '''

        vocab_file = os.path.join(bert_weight_path_base,'vocab.txt')
        bert_config_file = os.path.join(bert_weight_path_base,'bert_config.json')
        init_checkpoint = os.path.join(bert_weight_path_base,'bert_model.ckpt')
        common_command = " --vocab_file="+vocab_file+" --bert_config_file="+bert_config_file+" --init_checkpoint="+init_checkpoint+" --layers=-1 --max_seq_length=510 --batch_size=1"



        input_poem = "sub_poem.txt"
        input_poem = os.path.join(self.base_dir_for_save,input_poem)

        output_poem_vec = "bert_poem_vec.json"
        output_poem_vec = os.path.join(self.base_dir_for_save,output_poem_vec)
        poem_command = "python extract_features.py --input_file="+input_poem+" --output_file="+output_poem_vec+common_command
        os.system(poem_command)

        input_song = "sub_song.txt"
        input_song = os.path.join(self.base_dir_for_save, input_song)
        output_song_vec = "bert_song_vec.json"
        output_song_vec = os.path.join(self.base_dir_for_save, output_song_vec)
        song_command = "python extract_features.py --input_file=" + input_song + " --output_file=" + output_song_vec + common_command
        os.system(song_command)

    def MatchPoemSong(self,mode = "normal"):
        '''
        1。 先对歌曲和散文排序 ，以match_score_with_topic为关键字
        2. 再取前100-200的歌曲和散文
        3. 从歌曲对象--》歌曲歌词，散文对象--》散文文字
        4.得到从句子到向量的转换
        :return:
        '''
        self.sub_songs.sort(key = lambda song:song['match_score_with_topic'],reverse=True)
        self.sub_poems.sort(key = lambda poem:poem['match_score_with_topic'],reverse=True)
        self.Prepro()
        if self.mode == "gensim":
            self.VecPoem = self.Text2Vec(self.TextPoem)
            self.VecSong = self.Text2Vec(self.TextSong)
        elif self.mode == "bert":
            self.getBertVec()
        poemVecs = np.array(self.VecPoem)
        songVecs = np.array(self.VecSong)
        scoreMatrix = poemVecs.dot(songVecs.T)  # scoreMatrix[i,j] 第i首诗和第j首歌之间的相似度
        pickle.dump(scoreMatrix, open(os.path.join(self.base_dir_for_save,"similarityMatrix.pkl"), 'wb'))
        scoreMatrix = pickle.load(open(os.path.join(self.base_dir_for_save,"similarityMatrix.pkl"), "rb"))
        if mode == "normal":
            poemBestSong = scoreMatrix.argmax(axis=1)  # 针对每诗，最匹配的歌曲
            match_score = scoreMatrix.max(axis=1)  # 对应的分数
            songMatchedPoems = dict()  # key：song_index ， value：(poem_index,score)
            # 防重复
            for poem_index, (song_index, score) in enumerate(zip(poemBestSong, match_score)):
                if song_index not in songMatchedPoems:
                    songMatchedPoems[song_index] = (poem_index, score)
                else:
                    if score > songMatchedPoems[song_index][1]:
                        songMatchedPoems[song_index] = (poem_index, score)
            print("节目数：", len(songMatchedPoems))
            self.WriteToFile(songMatchedPoems)
        else:
            with open(os.path.join(self.base_dir_for_save, "song_poem_200_top10_new.txt"), "w", encoding="utf8") as fout:
                for poem_index , song_scores in enumerate(scoreMatrix):
                    poem = self.sub_poems[poem_index]
                    fout.write("poem:\t")
                    fout.write(''.join(poem['content'].split()))  # 以后还会用到poem_id这个字段，以便给一整首poem配歌
                    fout.write("\n")
                    topk_song_indexs = song_scores.argsort()[::-1][:10]
                    for si, song_index in enumerate( topk_song_indexs ):
                        song = self.sub_songs[song_index]
                        fout.write("song"+str(si)+":\t")
                        fout.write("name:")
                        fout.write(song['music_name']['ori_name'])
                        fout.write("\t")
                        fout.write("singer:")
                        fout.write(song['singer_info'][0]['ori_name'])
                        fout.write("\n")
                        fout.write("lyric:")
                        temp = song['lyric']['ori_lyric']
                        temp = '#'.join(temp.split())
                        fout.write(temp)
                        fout.write("\n")
                    fout.write("\n\n")

    def WriteToFile(self,songMatchedPoems):
        with open(os.path.join(self.base_dir_for_save,"song_poem_top10.txt"), "w", encoding="utf8") as fout:
            for song_index, (poem_index, score) in songMatchedPoems.items():
                song = self.sub_songs[song_index]
                poem = self.sub_poems[poem_index]
                fout.write("poem:\t")
                fout.write(''.join(poem['content'].split()))  #以后还会用到poem_id这个字段，以便给一整首poem配歌
                fout.write("\n")
                fout.write("song:\t")
                fout.write("name:")
                fout.write(song['music_name']['ori_name'])
                fout.write("\t")
                fout.write("singer:")
                fout.write(song['singer_info'][0]['ori_name'])
                fout.write("\n")
                fout.write("lyric:")
                temp = song['lyric']['ori_lyric']
                temp = '#'.join(temp.split())
                fout.write("\n\n")

    def MatchPoemSong2(self,getP_before=True):
        '''
        1。 先对歌曲和散文排序 ，以match_score_with_topic为关键字
        2. 再取前100-200的歌曲和散文
        3. 从歌曲对象--》歌曲歌词，散文对象--》散文文字
        4.得到从句子到向量的转换
        :return:
        '''
        self.sub_songs.sort(key = lambda song:song['match_score_with_topic'],reverse=True)
        self.sub_poems.sort(key = lambda poem:poem['match_score_with_topic'],reverse=True)
        print(len(self.sub_poems))
        print(len(self.sub_songs))
        self.Prepro()
        if self.mode == "gensim":
            self.VecPoem = self.Text2Vec(self.TextPoem)
            self.VecSong = self.Text2Vec(self.TextSong)
        elif self.mode == "bert":
            self.getBertVec()
        poemVecs = np.array(self.VecPoem)
        songVecs = np.array(self.VecSong)
        scoreMatrix = poemVecs.dot(songVecs.T)  # scoreMatrix[i,j] 第i首诗和第j首歌之间的相似度
        pickle.dump(scoreMatrix, open(os.path.join(self.base_dir_for_save,"similarityMatrix.pkl"), 'wb'))
        scoreMatrix = pickle.load(open(os.path.join(self.base_dir_for_save,"similarityMatrix.pkl"), "rb"))

        if getP_before:
            Poem2subp = {}
            for poem_index, scores in enumerate(scoreMatrix):
                poem = self.sub_poems[poem_index]
                poem_id, para_id = poem['poem_id'], poem["para_id"]
                if poem_id not in Poem2subp:
                    Poem2subp[poem_id] = []
                Poem2subp[poem_id].append((poem_index,para_id,scores))

            temp = {}
            for key , value in Poem2subp.items():
                if len(value) >=3:
                   temp[key] = value
            Poem2subp = temp
            del temp


            selected_song_ids = []
            candidated_shows = []
            for poem_id, values in Poem2subp.items():
                #先将value按照 scores中的大小排序
                length = len(values)
                one_candidated_show = []
                while length:
                    length -=1
                    values = sorted(values , key = lambda ele : ele[2].max() , reverse=True)
                    cur_value = values[0]
                    values.pop(0)
                    poem_index, para_id, scores = cur_value
                    best_song_indx = scores.argmax()
                    selected_song_ids.append(best_song_indx)
                    one_candidated_show.append((poem_id,para_id,poem_index,best_song_indx,scores[best_song_indx]))
                    #清除这首歌的index
                    for i in range(len(values)):
                        values[i] = (values[i][0],values[i][1] , np.delete(values[i][2],best_song_indx) )
                        #values[i][2] =  np.delete(values[i][2],best_song_indx) #是tuple，不能写
                candidated_shows.append(one_candidated_show)

            #算各个秀的总分
            def my_get_sum(one_triple):
                tot = 0
                for i in range(3):
                    tot += one_triple[i][-1]
                return tot

            for one_candidated_show in candidated_shows:
                one_candidated_show.sort(key = lambda ele : ele[1])#按照para_id排序

            shows = []
            for one_candidated_show in candidated_shows:
                '''
                one_candidated_show 中可能有超过3个匹配pair的情况
                '''
                if len(one_candidated_show) > 3:
                    for i in range(len(one_candidated_show) - 3 + 1):
                        one_show = {}
                        tot = my_get_sum(one_candidated_show[i:i+3])
                        one_show['score']=tot
                        one_show['content']=one_candidated_show[i:i+3]
                        shows.append(one_show)
                else:
                    one_show = {}
                    tot = my_get_sum(one_candidated_show[0 : 3])
                    one_show['score'] = tot
                    one_show['content'] = one_candidated_show[0 : 3]
                    shows.append(one_show)
            shows.sort(key = lambda ele : ele['score'],reverse=True)
            self.WriteToFile2(shows)

        else:
            poemBestSong = scoreMatrix.argmax(axis=1)  # 针对每篇散文，最匹配的歌曲
            match_score = scoreMatrix.max(axis=1)  # 对应的分数
            songMatchedPoems = dict()  # key：song_index ， value：(poem_index,score)
            # 防重复
            for poem_index, (song_index, score) in enumerate(zip(poemBestSong, match_score)):
                if song_index not in songMatchedPoems:
                    songMatchedPoems[song_index] = (poem_index, score)
                else:
                    if score > songMatchedPoems[song_index][1]:
                        songMatchedPoems[song_index] = (poem_index, score)
            #得到songMatchedPoems后,寻找有同样的poem_id的sub_poem
            Poem2subp ={}
            for song_index, (poem_index, score) in songMatchedPoems.items():
                poem = self.sub_poems[poem_index]
                poem_id , para_id = poem['poem_id'],poem["para_id"]
                if poem_id not in Poem2subp:
                    Poem2subp[poem_id]=[]
                Poem2subp[poem_id].append((para_id,song_index,score))
            for key,value in Poem2subp.items():
                # print(len(value))
                if len(value) >= 3:
                    print("yes")
                    print("ele = ",value)

    def WriteToFile2(self,shows):
        with open(os.path.join(self.base_dir_for_save,"song_poem_version3.txt"), "w", encoding="utf8") as fout:
            for one_show in shows:
                for item in one_show['content']:
                    (poem_id, para_id, poem_index, best_song_indx, score_of_match) = item
                    poem = self.sub_poems[poem_index]
                    song = self.sub_songs[best_song_indx]
                    fout.write("poem:\t")
                    fout.write(''.join(poem['content'].split()))  #以后还会用到poem_id这个字段，以便给一整首poem配歌
                    fout.write("\n")
                    fout.write("song:\t")
                    fout.write("name:")
                    fout.write(song['music_name']['ori_name'])
                    fout.write("\t")
                    fout.write("singer:")
                    fout.write(song['singer_info'][0]['ori_name'])
                    fout.write("\n")
                    fout.write("lyric:")
                    temp = song['lyric']['ori_lyric']
                    temp = '#'.join(temp.split())
                    #fout.write(temp)
                    fout.write("\n\n")
                fout.write("*"*100)
                fout.write("\n\n")

    def WriteSelectedPoemToFileForLook(self,sub_poems,out_path):
        os.makedirs('\\'.join(out_path.split('\\')[:-1]),exist_ok=True)
        for one_prose in sub_poems:
            one_prose['content'] = ''.join(one_prose['content'].split())
        json.dump(sub_poems,open(out_path,"w",encoding="utf-8"),ensure_ascii=False)

    def save(self):
        pass

    def load(self):
        pass

    def forward(self,mode="gensim"):
        self.mode = mode
        self.getPoem()
        pickle.dump(self.sub_poems,open(os.path.join(self.base_dir_for_save,"sub_poems"+str(self.opt['poem_threshold'])+".pkl"),"wb"))
        json.dump(self.sub_poems,open(os.path.join(self.base_dir_for_save,"sub_poems"+str(self.opt['poem_threshold'])+".json"),"w",encoding="utf-8"),ensure_ascii=False)
        self.WriteSelectedPoemToFileForLook(self.sub_poems, "ForLook\\sub_poems"+str(self.opt['poem_threshold'])+".json")
        print("len(sub_poems) = ",len(self.sub_poems))
        exit(90)
        # self.getSong()
        # pickle.dump(self.sub_songs,open(os.path.join(self.base_dir_for_save,"sub_songs.pkl"),'wb'))
        self.sub_songs = pickle.load(open(os.path.join(self.base_dir_for_save,"sub_songs.pkl"),"rb"))
        #self.sub_poems = pickle.load(open(os.path.join(self.base_dir_for_save,"sub_poems.pkl"),"rb"))
        self.sub_poems = json.load(open(os.path.join(self.base_dir_for_save,"sub_poems"+str(self.opt['poem_threshold'])+".json"),"r",encoding="utf-8"))

        print("len(self.sub_poems)",len(self.sub_poems))  # 211
        print("len(self.sub_songs)",len(self.sub_songs))  # 201
        #self.MatchPoemSong(mode = "lihai")
        self.MatchPoemSong2(getP_before=True)

    def T_forward(self):
        # self.getSong()
        # exit(90)
        #self.sub_songs = json.load(open(os.path.join(self.base_dir_for_save, "sub_songs221.json"), "r" , encoding="utf-8"))
        self.sub_songs = pickle.load(open(os.path.join(self.base_dir_for_save, "sub_songs471.pkl"), "rb"))
        # self.sub_poems = pickle.load(open(os.path.join(self.base_dir_for_save, "sub_poems.pkl"), "rb"))
        # print("len(self.sub_poems) = ", len(self.sub_poems))  # 211
        print("len(self.sub_songs) = ", len(self.sub_songs))  # 201
        # self.sub_songs.sort(key=lambda song: song['match_score_with_topic'], reverse=True)
        # self.sub_poems.sort(key=lambda poem: poem['match_score_with_topic'], reverse=True)

        # File_Poems = []
        # for poem in self.sub_poems:
        #     if poem['match_score_with_topic'] > 0.55:
        #         File_Poems.append(poem)
        #     else:
        #         break
        # with open(os.path.join(self.base_dir_for_save, "T_poem_055.json"), "w", encoding="utf-8") as fout:
        #     json.dump(File_Poems,fout,ensure_ascii=False)

        # with open(os.path.join(self.base_dir_for_save,"T_poem_055.json"),"w",encoding="utf-8") as fout:
        #     json.dump(self.sub_poems,fout,ensure_ascii=False)
        # with open(os.path.join(self.base_dir_for_save,"T_poem_100.json"),"w",encoding="utf-8") as fout:
        #     json.dump(self.sub_poems[:104],fout,ensure_ascii=False)



        File_Songs = []
        for song in self.sub_songs:
            File_Songs.append(
                {'name':song['music_name']['ori_name'],
                 'singer':song['singer_info'][0]['ori_name'],
                 'lyric':song['lyric']['ori_lyric'],
                 'match_score_with_topic':song['match_score_with_topic']
                }
            )
        with open(os.path.join(self.base_dir_for_save, "T_song_050.json"), "w", encoding="utf-8") as fout:
            json.dump(File_Songs,fout,ensure_ascii=False)

class MatchSeggedPoemSong(MatchPoemSong):
    def __init__(self ,opt, poem_file:str , song_file:str ,
                 out_file :str, keywords:List[str] ,
                 base_dir_for_save = r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem_Song',
                 save_dir_for_songVec = "songVec.pkl",
                 save_dir_for_poemVec = "poemVec.pkl",
                 ):

        #MatchPoemSong.__init__(self , opt , poem_file , song_file , keywords , base_dir_for_save , save_dir_for_songVec , save_dir_for_poemVec)
        super(MatchSeggedPoemSong,self).__init__(opt , poem_file , song_file , keywords , base_dir_for_save , save_dir_for_songVec , save_dir_for_poemVec)
        self.poem_file = poem_file
        self.song_file = song_file
        self.opt = opt
        self.base_dir_for_save = base_dir_for_save
        self.save_dir_for_songVec = save_dir_for_songVec
        self.save_dir_for_poemVec = save_dir_for_poemVec
        self.keywords = keywords
        print("MatchSeggedPoemSong.__init__(): keywords = ",self.keywords)
        self.poems = []
        self.sub_poems = []
        self.songs = []
        self.sub_songs = []
        self.songVecs = []
        self.poemVecs = []
        self.model = g.Doc2Vec.load(gensim_weight_path)
        self.out_file = out_file

    def SatisfyBoundConstrain(self,num_bound):
        if self.opt['lower_bound'] and self.opt['upper_bound']:
            if self.opt['lower_bound'] <= num_bound <= self.opt['upper_bound']:
                return True
        else:
            if self.opt['lower_bound'] == num_bound:
                return True
        return False

    def getSeggedPoem(self):
        with open(self.poem_file,"r",encoding='utf8') as fin:
            self.poems = json.load(fin)
        print("getSeggedPoem len(self.poems) = ",len(self.poems))
        self.keywords_count = Counter(self.keywords)
        self.segged_poems = []
        for poem_dict in self.poems:
            poem_id = poem_dict['poem_id']
            for para_id , para_dict in enumerate(poem_dict['paras']):
                score =  self.filterPoem(para_dict)
                if poem_dict["poem_class"] in self.keywords:
                    score += 10 + self.opt['poem_threshold']
                if score < self.opt['poem_threshold']: #0.001
                    continue
                if ('seg_point' not in para_dict) or ( not ( self.SatisfyBoundConstrain( len(para_dict['seg_point'])))):
                    continue
                seg_point = para_dict['seg_point']
                print("getSeggedPoem seg_point = ",seg_point)
                title = para_dict['fencied_para_title']
                title = ' '.join(title).strip()
                content = []
                one_seg = []
                one_seg_original = []
                for  i , para_c in enumerate(para_dict['fencied_para_content']):
                    one_seg.append(' '.join(para_c))
                    one_seg_original.append(''.join(para_c))
                    if i in seg_point:
                        content.append({"text":' '.join(one_seg),
                                        'original_text':'\n'.join(one_seg_original)})
                        one_seg = []
                        one_seg_original = []
                        seg_point.remove(i)
                temp = {'poem_id':poem_id,"para_id":para_id,'match_score_with_topic':score,"content":content,"url":poem_dict['url']}
                self.segged_poems.append(temp)
        self.segged_poems.sort(key=lambda poem: poem['match_score_with_topic'], reverse=True)

    def ProseToVec(self):
        '''
        segged_poems with the type of list [], each element is a dict,
        element['content']
          = [{text':'first Segmentation'},{text':'first Segmentation'},....]
        '''
        for one_prose_dict in self.segged_poems:
            for prose in one_prose_dict['content']:
                prose["vec"] = self.model.infer_vector(prose['text'].split())

    def SongToVec(self):
        for one_song in self.sub_songs:
            one_song['vec'] = self.model.infer_vector(one_song['lyric']['wb_lyric'])

    def SimScoreCal(self):
        '''计算各个，跟所有的歌曲的相似度'''
        songVecs = np.array([one_song['vec'] for one_song in self.sub_songs]).T
        self.shows = []
        for one_prose_dict in self.segged_poems:
            before_best_index = set()
            sum_score = 0
            poem_song_match_threshold = self.opt['poem_song_match_threshold']
            for one_segment in one_prose_dict['content']:
                match_scores = np.dot(one_segment['vec'], songVecs)
                best_index = np.argmax(match_scores)
                while best_index in before_best_index:
                    match_scores[best_index] = -1
                    best_index = np.argmax(match_scores)
                best_score = match_scores[best_index]
                if best_score < poem_song_match_threshold:
                    break
                one_segment['best_song_index'] = best_index
                before_best_index.add(one_segment['best_song_index'])
                one_segment['best_song_score'] = best_score
                sum_score += best_score
                one_segment['best_match_song'] = self.sub_songs[best_index]
            one_prose_dict['show_score'] = sum_score / len(one_prose_dict['content'])
            self.shows.append(one_prose_dict)
        self.shows.sort(key = lambda t : t['show_score'])

    def WriteToFile(self):
        if os.path.exists(self.out_file):
            temp = self.out_file.split('.')
            temp[0] += time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.out_file = '.'.join(temp)
        with open(self.out_file,"w",encoding="utf-8") as fout:
            for show in self.shows:
                if "show_score" in show:
                    fout.write("\nshow score:"+str(show['show_score']))
                for one_segment in show['content']:
                    fout.write("\nProse:\n")
                    fout.write(one_segment['original_text'])
                    fout.write("\nSong:")
                    fout.write("\nmusic name:"+one_segment['best_match_song']['music_name']['format_name'])
                    fout.write("\nsinger name:"+one_segment['best_match_song']['singer_info'][0]['format_name'])
                    fout.write("\nlyrics : "+one_segment['best_match_song']['lyric']['ori_lyric'])
                    fout.write("="*30)
                fout.write("\n\n"+"*"*10+"one new show"+"*"*10)

    def forward(self,mode="gensim"):
        # self.mode = mode
        self.getSeggedPoem()
        # self.getSong()
        #TODO for faster run
        pickle.dump(self.segged_poems,open(os.path.join(self.base_dir_for_save,"segged_poems"+str(self.opt['poem_threshold'])+".pkl"),"wb"))
        json.dump(self.segged_poems,open(os.path.join(self.base_dir_for_save,"segged_poems"+str(self.opt['poem_threshold'])+".json"),"w",encoding="utf-8"),ensure_ascii=False)
        self.sub_songs = pickle.load(open(os.path.join(self.base_dir_for_save,"sub_songs.pkl"),"rb"))
        self.segged_poems = json.load(open(os.path.join(self.base_dir_for_save,"segged_poems"+str(self.opt['poem_threshold'])+".json"),"r",encoding="utf-8"))
        print("len(self.segged_poems) = ",len(self.segged_poems))  # 211
        print("len(self.sub_songs) = ",len(self.sub_songs))  # 201
        self.sub_songs.sort(key=lambda song: song['match_score_with_topic'], reverse=True)
        self.segged_poems.sort(key=lambda poem: poem['match_score_with_topic'], reverse=True)
        self.ProseToVec()
        self.SongToVec()
        self.SimScoreCal()
        self.WriteToFile()


class SearchSong:
    def __init__(self , query_web_site =  "http://10.190.178.145:8887/?"):
        self.query_web_site = query_web_site
        self.keywords = "keywords="
        self.keyword_weight = "keyword_weight="
        self.top_k = "top_k="
        self.lim = "lim="
        self.use_syn="use_syn="

    def Search(self,keywords = ['爱情'] , top_k=20, lim = 20 , keyword_weight=[1.0], use_syn=0):
        assert len(keywords) == len(keyword_weight)
        weight_sum = sum(keyword_weight)
        keyword_weight = [ str(round(ele/weight_sum,2)) for ele in keyword_weight]
        # keyword_weight = list(map(str,keyword_weight))
        query = [self.keywords+','.join(keywords),
                 self.top_k+str(top_k),
                 self.lim+str(lim),
                 self.keyword_weight+','.join(keyword_weight),
                 self.use_syn+str(use_syn)]
        query='&'.join(query)
        query = self.query_web_site + query
        try:
            List = requests.get(query).json()
        except:
            print("Error in request.")
            print("query = ",query)
            exit(78)
            return None
        return List





class PoemMatchSong(MatchSeggedPoemSong):
    def __init__(self ,opt, poem_file:str , song_file:str ,
                 keywords:List[str] ,
                 out_file: str,
                 base_dir_for_save = r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem_Song',
                 save_dir_for_songVec = "songVec.pkl",
                 save_dir_for_poemVec = "poemVec.pkl",
                 idf_path = None,
                 additional_key_words_path = None,
                 out_txt_dir = None):
        super(MatchSeggedPoemSong,self).__init__(opt , poem_file , song_file , keywords , base_dir_for_save , save_dir_for_songVec , save_dir_for_poemVec)
        self.base_dir_for_save = base_dir_for_save
        print("PoemMatchSong.__init__(): keywords = ",self.keywords)
        self.out_file = out_file
        self.idf_path = idf_path
        if self.idf_path is not None:
            analyse.set_idf_path(self.idf_path)
        self.additional_key_words = dict()
        if additional_key_words_path is not None:
            self.additional_key_words = self.load_additional_key_words(additional_key_words_path)
        self.modelSearchSong = SearchSong()
        self.out_txt_dir = out_txt_dir #写到txt中供yaml调用


    def load_additional_key_words(self,path,max_rows=200):
        wb = openpyxl.load_workbook(path)
        sheet_names = wb.get_sheet_names()
        res = dict()
        for sheet_name in sheet_names:
            sheet = wb.get_sheet_by_name(sheet_name)
            for i, row in enumerate(sheet.rows):
                if i == max_rows:
                    break
                if i == 0:
                    continue
                word = row[0].value
                weight = float(row[1].value)
                res[word]=weight
        return res

    def ExtractKeyWordForSeg(self):
        '''
        segged_poems with the type of list [], each element is a dict,
        element['content'] = [{text':'first Segmentation'},{text':'first Segmentation'},....]
        '''
        for one_prose_dict in self.segged_poems:
            for prose in one_prose_dict['content']:
                all_content = prose['original_text']
                # print("all_content = ",all_content)
                prose["key_words"] = analyse.textrank(all_content, topK=10, withWeight=True, allowPOS=('ns', 'n', 'vn','v'))#'v' #ns地名；n名词；vn名动词，比如思索；v动词
                prose["idf_key_words"] = analyse.extract_tags(all_content, topK=10, withWeight=True, allowPOS=('ns', 'n', 'vn','v'))#'v' #ns地名；n名词；vn名动词，比如思索；v动词
                prose["key_words"] = [[word_weight[0],word_weight[1] + self.additional_key_words[ word_weight[0] ] ] if word_weight[0] in self.additional_key_words.keys() else [ word_weight[0],word_weight[1] ]   for word_weight in prose["key_words"] ]
                prose["idf_key_words"] = [[word_weight[0],word_weight[1] + self.additional_key_words[ word_weight[0] ] ] if word_weight[0] in self.additional_key_words.keys() else [ word_weight[0],word_weight[1] ]   for word_weight in prose["idf_key_words"] ]
                prose["key_words"].sort(key = lambda t:t[1],reverse = True)
                prose["idf_key_words"].sort(key = lambda t:t[1],reverse = True)

    def findHighestScore(self,ChosenSongList,SongsUsed):
        '''
        :param ChosenSongList: List[Dict()]  具体的格式和内容，根据楼林辉的接口来定
        :param SongsUsed: 已经使用过的列表
        :return:
        '''
        Flag = False
        for ele in ChosenSongList:
            artist_name = ele["artist"]['name']
            song_name = ele["name"]
            if (artist_name,song_name) not in SongsUsed:
                SongsUsed.add((artist_name,song_name))
                Flag = True
                return Flag , (artist_name,song_name)
        return Flag , ("NULL","NULL")


    def findBestSongsForThisProse(self):
        '''
        segged_poems with the type of list [], each element is a dict,
        element['content'] = [{text':'first Segmentation'},{text':'first Segmentation'},....]
        '''
        self.shows = []
        for one_prose_dict in self.segged_poems:
            songs_used = set()
            '''
            one_prose_dict 中包含多个seg，每个seg会对应召回很多首歌曲。一个one_prose_dict即一个节目，一个节目中，有同样名字的歌曲是不合适的。
            '''
            Flag = True
            for prose in one_prose_dict['content']:
                keywords = [ele[0] for ele in  prose["idf_key_words"][:3]]
                keyword_weight = [ele[1] for ele in prose["idf_key_words"][:3]]
                flag = False
                top_k = 10
                lim = 10
                while not flag:
                    prose["chosen_songs"] = self.modelSearchSong.Search(keywords=keywords,keyword_weight=keyword_weight,top_k=top_k,lim=lim)
                    if prose["chosen_songs"] is None:
                        Flag = False
                        break
                    flag,(artist_name,song_name) = self.findHighestScore(prose['chosen_songs'],songs_used)
                    if flag:
                        prose["best_song"] = (artist_name,song_name)
                        break
                    else:
                        top_k *= 2
                        lim *= 2
                        if top_k > 100:
                            prose["best_song"] = ("NULL","NULL")
                            Flag = False
                            break
            if Flag:
                self.shows.append(one_prose_dict)

    def WriteToFile(self):
        if os.path.exists(self.out_file):
            temp = self.out_file.split('.')
            temp[0] += time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
            self.out_file = '.'.join(temp)
        with open(self.out_file,"w",encoding="utf-8") as fout:
            for show in self.shows:
                for one_segment in show['content']:
                    fout.write("\nProse:\n")
                    fout.write(one_segment['original_text'])
                    fout.write("\nSong:")
                    fout.write("\nmusic name:"+one_segment['best_song'][1])
                    fout.write("\nsinger name:"+one_segment['best_song'][0]+"\n")
                    fout.write("="*30)
                fout.write("\n\n"+"*"*10+"one new show"+"*"*10)

    def WriteToTxt(self):
        '''
        :return:
        '''
        out_txt_dir = self.out_txt_dir
        if os.path.exists(self.out_txt_dir):
            out_txt_dir = self.out_txt_dir + time.strftime("%Y-%m-%d_%H-%M-%S", time.localtime())
        os.makedirs(out_txt_dir,exist_ok=True)
        os.chdir(out_txt_dir)
        with open("data_"+str(self.opt['lower_bound'])+".txt","w",encoding="utf-8") as fout:
            for index, show in enumerate( self.shows):
                out_content = [str(index)]
                for i,one_seg in enumerate( show['content']):
                    original_text = one_seg['original_text'].split('\n')
                    original_text = " ".join(original_text)
                    out_content.append("text"+str(i+1)+":"+original_text)
                    out_content.append("name"+str(i+1)+":"+one_seg['best_song'][1])
                    out_content.append("singer"+str(i+1)+":"+one_seg['best_song'][0])
                out_content = '\t'.join(out_content)
                fout.write(out_content+"\n")




    def forward(self):
        self.getSeggedPoem()
        # TODO for faster run
        pickle.dump(self.segged_poems, open(
            os.path.join(self.base_dir_for_save, "segged_poems" + str(self.opt['poem_threshold']) + ".pkl"), "wb"))
        json.dump(self.segged_poems,
                  open(os.path.join(self.base_dir_for_save, "segged_poems" + str(self.opt['poem_threshold']) + ".json"),
                       "w", encoding="utf-8"), ensure_ascii=False)
        # TODO for selk.segged_poems 中包含这选出来的散文
        print("PoemMatchSong.forward():len(self.segged_poems) = ", len(self.segged_poems))
        self.segged_poems.sort(key=lambda poem: poem['match_score_with_topic'], reverse=True)
        self.ExtractKeyWordForSeg()
        self.findBestSongsForThisProse()
        self.WriteToFile()
        self.WriteToTxt()




if __name__ == "__main__":
    # T1()
    # exit(90)
    poem_song_matcher = MatchSeggedPoemSong(opt=opt,
                                      poem_file=os.path.join('E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem',
                                                             "seged_poem.json"),
                                            song_file=os.path.join('E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Song',
                                                                   "song4.pkl", ),
                                            out_file='E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem_Song\\seggedProseSong.txt',
                                            keywords=['夜晚', '深夜', '寂静', '安眠', '星空', '平静', '喧嚣', '静', '夜色', '月亮', "失眠"])
    poem_song_matcher.forward()
