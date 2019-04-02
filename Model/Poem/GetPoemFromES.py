# -*- coding:utf-8 -*-
import re
import os
import pickle
import json
from elasticsearch import Elasticsearch
from collections import Counter

ProjectPath = "FirstDayOnMS2"
abspath = os.path.abspath(os.getcwd())
abspath = abspath.split(ProjectPath)
prefix_path = os.path.join(abspath[0],ProjectPath)


class GetProse:
    def __init__(self,target = ['http://chitchat-index-int.eastasia.cloudapp.azure.com:19200/'],
                 http_auth = ('esuser', 'Kibana123!'),
                 port=9200, timeout=50000,
                 base_save_dir = os.path.join(prefix_path, "Data\\Poem"),
                 out_file_name = "poem_es.json"
                 ):
        super(GetProse,self).__init__()
        self.target = target
        self.http_auth = http_auth
        self.port = port
        self.timeout = timeout
        self.es = Elasticsearch(target,http_auth= http_auth,port=port,timeout=timeout)
        self.base_save_dir = base_save_dir
        self.out_file_name = out_file_name
        print("connect succeed!!!")

    def TurnBack(self,one_prose):
        '''存入es之前做了一些修改，以便使得格式上符合es的要求和检索的需要
        现在需要把这些再转换回来'''
        for one_para in one_prose["paras"]:
            temp = one_para["key_words"]
            one_para["key_words"] = []
            for one_pair in temp:
                one_para["key_words"].append([one_pair["key_words"], one_pair["weight"]])

            temp = one_para["idf_key_words"]
            one_para["idf_key_words"] = []
            for one_pair in temp:
                one_para["idf_key_words"].append([one_pair['key_words'], one_pair['weight']])

            temp = one_para["fencied_para_content"]
            one_para["fencied_para_content"] = []
            for one_fenci_para in temp:
                one_para["fencied_para_content"].append(one_fenci_para.split())

            one_para["fencied_para_title"] = one_para["fencied_para_title"].split()
        return one_prose

    def crawlPoem(self,index = 'prose_segged_by_algo/',doc_type="prose", key_words = ['爱情']):
        body = {
              "query": {
               "filtered":{
                "query":{
                    "match_all":{}
                }
                ,
                "filter":{
                    "or":[
                            {"match":{"paras.para_title":' '.join(key_words)}},
                            {"match":{"paras.fencied_para_content":' '.join(key_words)}},
                            {"match":{"poem_title":' '.join(key_words)}}
                        ]
                }
               }
              }
            }
        page = self.es.search(
            index=index,
            doc_type=doc_type,
            scroll='2m',
            # search_type='scan',
            size=1000,
            body=body
        )
        sid = page['_scroll_id']
        scroll_size = page['hits']['total']
        print("scroll_size = ",scroll_size)
        # Start scrolling
        self.ResultList = []
        self.ResultNum = 0
        self.pklName = 0
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
                for i, ele in enumerate(page['hits']['hits']):
                    one_poem = ele["_source"]
                    one_poem = self.TurnBack(one_poem)
                    self.ResultList.append(one_poem)
                    self.ResultNum += 1
                    print("in while in for self.ResultNum = ",self.ResultNum)
            except Exception as err:
                print('Exception: ', err)
                print("先保存下来")
                print("ResultNum = ", self.ResultNum)
                json.dump(self.ResultList,open(os.path.join(self.base_save_dir,"poem"+str(self.pklName) + ".json"),"w",encoding="utf-8"),ensure_ascii=False)
                self.ResultList = []
                self.ResultNum = 0
                self.pklName += 1
                print("save pkl succeed!")
                print("ResultNum = ", self.ResultNum)
                print("pklName = ", self.pklName)
            # if len(self.ResultList) >= 30:
            #     print("先保存下来")
            #     print("ResultNum = ", self.ResultNum)
            #     json.dump(self.ResultList,
            #               open(os.path.join(self.base_save_dir, "poem" + str(self.pklName) + ".json"), "w",
            #                    encoding="utf-8"), ensure_ascii=False)
            #     self.ResultList = []
            #     self.ResultNum = 0
            #     self.pklName += 1
            #     print("save pkl succeed!")
            #     print("ResultNum = ", self.ResultNum)
            #     print("pklName = ", self.pklName)
            #     break
        if self.ResultNum > 0:
            print("the ResultNum = ",self.ResultNum)
            json.dump(self.ResultList,
                     open(os.path.join(self.base_save_dir, "poem" + str(self.pklName) + ".json"), "w",
                          encoding="utf-8"), ensure_ascii=False)
            self.ResultList = []
            self.ResultNum = 0
            self.pklName += 1
            print("save pkl succeed!")
            print("ResultNum = ", self.ResultNum)
            print("pklName = ", self.pklName)
        res = []
        for temp_pklName in range(self.pklName):
            res.extend(json.load(open(os.path.join(self.base_save_dir, "poem" + str(temp_pklName) + ".json"), "r",
                          encoding="utf-8")))
        json.dump(res,
                  open(os.path.join(self.base_save_dir, self.out_file_name), "w",
                       encoding="utf-8"), ensure_ascii=False)
        for temp_pklName in range(self.pklName):
            os.remove(os.path.join(self.base_save_dir, "poem" + str(temp_pklName) + ".json"))


if __name__ == "__main__":
    pass