'''
将爬取到的，并清洗好的数据，存入到es数据库中
散文：prose
新建的index:prose_20190220
'''

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import os
from Poem.prose_mapping import custom_mapping

class StoreProse:
    def __init__(self,target = ['http://chitchat-index-int.eastasia.cloudapp.azure.com:19200/'],
                 http_auth = ('esuser','Kibana123!'),
                 port = 9200 , timeout = 50000,
                 ):
        super(StoreProse,self).__init__()
        self.es = Elasticsearch(target, http_auth = http_auth, port = port, timeout = timeout)
        print("connect succeed!!!")
        self.prose = None

    def getProse(self , path):
        with open(path,"r",encoding="utf-8") as fin:
            self.prose = json.load(fin)
        for one_prose in self.prose:
            for one_para in one_prose["paras"]:
                temp = one_para["key_words"]
                one_para["key_words"] = []
                for one_pair in temp:
                    one_para["key_words"].append({'key_words':one_pair[0],"weight":one_pair[1]})

                temp = one_para["idf_key_words"]
                one_para["idf_key_words"] = []
                for one_pair in temp:
                    one_para["idf_key_words"].append({'key_words': one_pair[0], "weight": one_pair[1]})

                temp = one_para["fencied_para_content"]
                one_para["fencied_para_content"] = []
                for one_fenci_para in temp:
                    one_fenci_para = ' '.join(one_fenci_para)
                    one_para["fencied_para_content"].append(one_fenci_para)

                one_para["fencied_para_title"] = ' '.join(one_para["fencied_para_title"])
        json.dump(self.prose, open("processed_poem_to_es.json", "w", encoding="utf-8"), ensure_ascii=False)

    def set_mapping(self,doc_type,target_index):
        mappings = {
                    "properties":
                                {
                                    "fencied_para_content":
                                        {
                                                "type":"string",
                                                "analyzer":"whitespace"
                                         },
                                    "fencied_para_title":
                                        {
                                            "type":"string",
                                            "analyzer":"whitespace"
                                        }
                                }
                   }
        self.es.indices.put_mapping(
            doc_type=doc_type,
            body = mappings,
            index = target_index
        )

    def storeProse(self,index_ = "prose_segged_by_algo",type_="prose"):
        result=self.es.indices.delete(index=index_)
        print("result = ",result)
        # exit(6)
        result=self.es.indices.create(index=index_)
        print("result = ",result)
        self.set_mapping(doc_type=type_,target_index=index_)
        print("over")
        # if self.prose is None:
        #     self.prose = json.load(open("processed_poem_to_es.json", "r", encoding="utf-8"))
        # print("len = ",len(self.prose))
        # data = self.prose[0]
        # print("data = ")
        # print(data)
        # resutlt = self.es.create(index="prose_segged_by_algo", doc_type='prose', id=1, body=data)
        # print(resutlt)
        # print("nice")
        # exit(8)
        ACTIONS = []
        for line in self.prose:
            action = {
                "_index":index_,
                "_type":type_,
                "_source":line
            }
            ACTIONS.append(action)
        success, _ = bulk(self.es,ACTIONS,index = index_, raise_on_error = True)
        print('Performed %d actions' % success)







if __name__ == "__main__":
    ProjectPath = "FirstDayOnMS2"
    abspath = os.path.abspath(os.getcwd())
    abspath = abspath.split(ProjectPath)
    prefix_path = os.path.join(abspath[0], ProjectPath)
    ST = StoreProse()
    ST.getProse(os.path.join(prefix_path,'Data\\Poem\\seged_poem_2019.json'))
    ST.storeProse()


