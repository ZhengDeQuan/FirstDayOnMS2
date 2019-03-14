# -*- coding:utf-8 -*-
import re
import os
import pickle
import json
from elasticsearch import Elasticsearch
from Song.commentProvider import CommentProvider
from Song.CheckCopyRight import music_search
from collections import Counter



class songSpider:
    def __init__(self,target = ['http://chitchat-index-int.eastasia.cloudapp.azure.com:19200/'],
                 http_auth = ('esuser', 'Kibana123!'),
                 port=9200, timeout=50000,
                 base_save_dir = 'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Song',
                 filter_by_comment = True,
                 pklName = 5):
        super(songSpider,self).__init__()
        self.target = target
        self.http_auth = http_auth
        self.port = port
        self.timeout = timeout
        self.es = Elasticsearch(target,http_auth= http_auth,port=port,timeout=timeout)
        print("connect succeed!!!")
        '''
        连接数据库
        '''
        self.ch_pattern = re.compile('[\u4e00-\u9fa5]+')
        self.ch_word_ratio = 0.7 #中文单词超过这个比例才算是中文歌
        '''
        判中文
        '''
        self.ResultList = []
        self.ResultNum = 0
        self.pklName = pklName
        self.base_save_dir = base_save_dir
        self.filter_by_comment = filter_by_comment
        self.sources_to_check = [music_search.MusicSource.MIGU]
        self.copyRightChecker = music_search.MusicSearch()

    def JudgeChinese(self,content,ratio = 0.7):
        content = content.split()
        '''
        最容易弄混的是日文歌，因为日文歌中可以有中文的汉字
        '''
        num_Chinese_word = 0
        num_nonChinese_word = 0
        for word in content:
            flag = True
            for ch in word:
                if not ('\u4e00' <= ch <= '\u9fa5'):
                    flag = False
                if flag is False:
                    break
            if flag is False:
                num_nonChinese_word += 1
            else:
                num_Chinese_word += 1
            isJapan = False
            for ch in word:
                if '\u3040' <= ch <= '\u309F' or '\u30A0' <= ch <= '\u30FF':
                    #日文平假名， 日文片假名
                    isJapan = True
                    break
            if isJapan:
                return False
        if num_nonChinese_word + num_Chinese_word == 0:
            return False
        if num_Chinese_word / (num_Chinese_word + num_nonChinese_word) > ratio:
            return True
        return False

    def JudgeCopyRight(self,song):
        '''
        已经利用can_play字段是否为true，在查询的时候就修改了
        '''
        song_name = song['music_name']['format_name']
        singers = []
        for sig in song['singer_info']:
            singers.append(sig['format_name'])
        try:
            has_copyright_sources = self.copyRightChecker.check_copyright(name=song_name, singers=singers,
                                                        sources_to_check=self.sources_to_check)
        except Exception("找版权的时候有问题的了"):
            return False
        if len(has_copyright_sources) > 0:  # 有版权
            return True
        return False


    def crawlSong(self,index = 'netease_music_merged_current/',doc_type="e_music"):
        # Initialize the scroll

        # body = {
        #     "query": {
        #         "match_all": {}
        #     }
        # }
        body = {
               "query" : {
                  "filtered" : {
                     "filter" : {
                        "bool" : {
                          "must_not":[{"match":{"lyric.wb_lyric": "pure music"}}],
                          "must" : [
                            {"exists" : [{ "field" : "lyric" }]},
                            {"nested": {
                                    "path": "tags",
                                    "query": {
                                      "exists": {
                                        "field": "tags"
                                      }
                                    }

                                }

                            },
                            { "match": { "can_play": "true" }}
                          ]
                       }
                     }
                  }
               },
               "_source": ["album_info", "lyric", "music_id", "music_name", "singer_info", "similar_musics", "tags",
                           "popularity"]
            }
        '''
        body里面一共有4个查询条件：
        1 lyric.web_lyric字段不等于pure music 确保有歌词
        2 lyric字段存在，确保有歌词
        3 tags字段存在，确保有tag
        4 can_play字段为true，确保有版权
        剩下需要在程序中判断的就是："歌曲是不是中文歌词的了"
        '''
        body_migu = {
               "query" : {
                  "bool" : {
                     "filter" : {
                        "bool" : {
                          "must_not":[{"match":{"lyric.wb_lyric": "pure music"}},
                                      {"match":{"lyric.wb_lyric":"暂 无 歌词"}},
                                      {"match":{"lyric.wb_lyric":"此 歌曲 为 没有 填词 的 纯 音乐 请 您 欣赏"}}],
                          "must" : [
                            {"exists" : { "field" : "lyric" }
                             },
                            {"nested": {
                                    "path": "tags",
                                    "query": {
                                      "exists": {
                                        "field": "tags"
                                      }
                                    }

                                }

                            },
                            { "match": { "can_play": "true" }}
                          ]
                       }
                     }
                  }
               },
               "_source": ["album_info", "lyric", "music_id", "music_name", "singer_info", "similar_musics", "tags",
                           "popularity"]
            }

        page = self.es.search(
            index=index,
            doc_type=doc_type,
            scroll='2m',
            # search_type='scan',
            size=1000,
            body=body_migu
        )
        sid = page['_scroll_id']
        scroll_size = page['hits']['total']
        print("scroll_size = ",scroll_size)


        # Start scrolling
        while (scroll_size > 0):
            print("Scrolling...")
            # Update the scroll ID
            sid = page['_scroll_id']
            # Get the number of results that we returned in the last scroll
            scroll_size = len(page['hits']['hits'])
            print("scroll size: " + str(scroll_size))
            # Do something with the obtained page
            try:
                page = self.es.scroll(scroll_id=sid, scroll='2m')
                #print("length = ",page['hits']['hits'])
                for i, ele in enumerate(page['hits']['hits']):  # type(page['hits']['hits']) == list , len(page['hits']['hits'])==1000
                    # for key in ele['_source']:  # _source 里面才有真正有用的东西 music_id , album_info, alias = [] , singer_info, music_name ,
                    #     print("key = ", key)
                    #     print("value = ", ele['_source'][key])
                    # exit(89)
                    # if i == 2:
                    #     exit(56789)
                    if len(ele['_source']['lyric']['wb_lyric']) > 0:
                        # print(ele['_source']['lyric']['ori_lyric'])#分段的，但是没有分词
                        # print('\n')
                        # print(ele['_source']['lyric']['notime_lyric'])#同上
                        # print('\n')
                        lyric = ele["_source"]['lyric']['wb_lyric']  # 分过词的，一定要用分过词的判断，否则，不公平，英文中一个单词有很多字符
                        # print('i= ', i, " len(page) = ", len(page['hits']['hits']), " lyric=", lyric[:30])
                        # if self.JudgeChinese(lyric) and self.JudgeCopyRight(ele["_source"]):
                        if self.JudgeChinese(lyric):

                            # 加入到最终的结果中
                            #print("Judge Chinese Succeed ",'i= ',i," len(page) = ",len(page['hits']['hits']) , " lyric=",lyric[:30])
                            # tempzhengquan = input()
                            ori_name, ori_singers, ori_album = ele["_source"]['music_name']['ori_name'],\
                                                               ele["_source"]['singer_info'][0]['ori_name'], \
                                                               None# ele["_source"]['album_info'][0]['ori_name']
                            temp = ele["_source"]
                            # comments , byReplied = CommentProvider.get_comments(ori_name, ori_singers, ori_album)#ori_name, ori_singers, ori_album
                            # if len(comments) == 0:
                            #     print("comments = = 0")
                            #     continue

                            # comments = Counter(comments)
                            # byReplied = Counter(byReplied)
                            # comments  = comments - byReplied
                            # temp["comments"] = comments
                            self.ResultList.append(temp)
                            self.ResultNum += 1
                            print("in while in for self.ResultNum = ",self.ResultNum)
                            # if len(self.ResultList) >= 30:
                            #     break
                                # print("ResultNum = ",self.ResultNum)
                                # pickle.dump(self.ResultList, open("song" + str(self.pklName) + ".pkl", "wb"))
                                # self.ResultList = []
                                # self.ResultNum = 0
                                # self.pklName += 1
                                # print("save pkl succeed!")
                                # print("ResultNum = ",self.ResultNum)
                                # print("pklName = ",self.pklName)


                #exit(678)
                #break#一个page之后结束
            except Exception("又发生了错误"):
                print("先保存下来")
                break

            # if len(self.ResultList) >= 30:
            #     break



        if self.ResultNum > 0:
            print("the ResultNum = ",self.ResultNum)
            if self.filter_by_comment:
                pickle.dump(self.ResultList, open(os.path.join(self.base_save_dir,"songWithComment" + str(self.pklName) + ".pkl"), "wb"))
                json.dump(self.ResultList,open(os.path.join(self.base_save_dir,"songWithComment"+str(self.pklName) + ".json"),"w",encoding="utf-8"),ensure_ascii=False)
            else:
                pickle.dump(self.ResultList, open(os.path.join(self.base_save_dir,"song" + str(self.pklName) + ".pkl"), "wb"))
                json.dump(self.ResultList,
                     open(os.path.join(self.base_save_dir, "song" + str(self.pklName) + ".json"), "w",
                          encoding="utf-8"), ensure_ascii=False)
            self.ResultList = []
            self.ResultNum = 0
            self.pklName += 1
            print("save pkl succeed!")
            print("ResultNum = ", self.ResultNum)
            print("pklName = ", self.pklName)


class MergeSongComment:
    def __init__(self,in_file,out_file="songComment.pkl",base_dir = 'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Song'):
        self.in_file = in_file
        self.out_file = out_file
        self.base_dir = base_dir
        self.songs = pickle.load(open(os.path.join(base_dir,in_file),"rb"))

    def MergeComment(self):
        self.ResultList = []
        for ele in self.songs:
            ori_name, ori_singers, ori_album = ele['music_name']['ori_name'], \
                                               ele['singer_info'][0]['ori_name'], \
                                               ele['album_info'][0]['ori_name']
            comments, byReplied = CommentProvider.get_comments(ori_name, ori_singers,
                                                               ori_album)  # ori_name, ori_singers, ori_album
            if len(comments) == 0:
                continue
            comments = Counter(comments)
            byReplied = Counter(byReplied)
            comments = comments - byReplied
            temp = ele
            temp["comments"] = comments
            self.ResultList.append(temp)
        pickle.dump(self.ResultList,open(os.path.join(self.base_dir,self.out_file),"wb"))

if __name__ == "__main__":
    # songClawer = songSpider(target=['http://corechat-usermemory-int.trafficmanager.net:19200/'],filter_by_comment=False,pklName=6)
    # songClawer.crawlSong(index='migu_music_merged_current')
    songClawer = songSpider()
    songClawer.crawlSong()
    # merger = MergeSongComment(in_file="song4.pkl")
    # merger.MergeComment()

    # a = ['', '', '']#ori_name, ori_singers, ori_album
    # b = [a, a, a, a, a]
    # comments, beReplied = CommentProvider.get_comments("", "", "")
    # print(len(comments))
    # for content in comments:
    #     print(content)
    #     print(len(content))
    # comments = list(CommentProvider.batch_get_comments(b))
    # print("*"*100)
    # print(len(comments))
    # for comment in comments:
    #     print(len(comment))
    #     for content in comment:
    #         print(content)
    #         print(len(content))