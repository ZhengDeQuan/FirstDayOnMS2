'''
将爬取到的，并清洗好的数据，存入到es数据库中
散文：prose
新建的index:prose_20190220
'''

from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import os
from prose_mapping import custom_mapping

class StoreProse:
    def __init__(self,target = ['http://chitchat-index-int.eastasia.cloudapp.azure.com:19200/'],
                 http_auth = ('esuser','Kibana123!'),
                 port = 9200 , timeout = 50000,
                 ):
        super(StoreProse,self).__init__()
        self.es = Elasticsearch(target, http_auth = http_auth, port = port, timeout = timeout)
        print("connect succeed!!!")

    def getProse(self , path):
        with open(path,"r",encoding="utf-8") as fin:
            self.prose = json.load(fin)
        for one_prose in self.prose:
            for one_para in one_prose["paras"]:
                temp = one_para["key_words"]
                one_para["key_words"] = []
                for one_pair in temp:
                    one_para["key_words"].append({'key_words':one_pair[0],"weight":one_pair[1]})
        json.dump(self.prose, open("processed_poem_to_es.json", "w", encoding="utf-8"), ensure_ascii=False)

    def set_mapping(self , es , index_ , type_):
        mapping = {
            type_:{
                "properties":{
                    "poem_id":{
                        "type":"integer"
                    },
                    "poem_title":{
                        "type": "text",  # title需要分词
                        "index": True,  # title需要分词
                        "analyzer": "smartcn",  # 使用smartcn分词器
                        "store": True,
                        "copy_to": "full_title"  # 全文保存到full_title
                    },
                    "origin_poem":{
                        "type": "text",  # poem需要分词
                        "index": True,  # poem需要分词
                        "analyzer": "smartcn",  # 使用smartcn分词器
                        "store": True,
                        "copy_to": "full_poem"  # 全文保存到full_poem
                    },
                    "poem_class":{
                        "type": "keyword"
                    },
                    "url":{
                        'type':"keywords"
                    },
                    "website_name":{
                        "type":"keyword"
                    },
                    "washed_poem":{
                        "type": "text",  # washed_poem需要分词
                        "index": True,  # washed_poem需要分词
                        "analyzer": "smartcn",  # 使用smartcn分词器
                        "store": True,
                        "copy_to": "full_washed_poem"  # 全文保存到full_washed_poem
                    },
                    "season":{        #es中任何一个字段都可以有多个值
                        "type":"keywords"
                    },
                    "festival":{
                        "type":"keywords"
                    },
                    "paras":{
                        "type":"nested",
                        "properties":{
                            "para_title":{
                                "type": "text",  # para_title需要分词
                                "index": True,  # para_title需要分词
                                "analyzer": "smartcn",  # 使用smartcn分词器
                                "store": True,
                                "copy_to": "full_para_title"  # 全文保存到full_para_title
                            },
                            "para_content":{ #es天然支持，一个字段，多个值
                                "type": "text",  # para_content需要分词
                                "index": True,  # para_content需要分词
                                "analyzer": "smartcn",  # 使用smartcn分词器
                                "store": True,
                                "copy_to": "full_para_content"  # 全文保存到full_para_content
                            },
                            "season":{
                                "type":"keywords"
                            },
                            "festival":{
                                "type":"keywords"
                            },
                            "key_words":{
                                "type":"nested",
                                "properties":{
                                    "key_word":{
                                        "type":"keywords"
                                    },
                                    "weight":{
                                        "type":"float"
                                    }
                                }
                            },
                            "fencied_para_title":{
                                "type":"keywords"
                            },
                            "fencied_para_content":{
                                "type":"keywords"
                            }
                        }
                    }

                }
            }
        }
        self.mapping = mapping
        return

    def set_mapping2(self):
        self.mapping = custom_mapping

    def storeProse(self,index_ = "prose_20190220",type_="prose"):

        # result=self.es.indices.delete(index=index_)
        # print("result = ",result)
        result=self.es.indices.create(index=index_)
        print("result = ",result)
        self.prose = json.load(open("processed_poem_to_es.json", "r", encoding="utf-8"))
        # data = self.prose[0]
        # resutlt = self.es.create(index="zhtest_20190220", doc_type='prose', id=1, body=data)
        # print(resutlt)
        ACTIONS = []
        for line in self.prose:
            action = {
                "_index":index_,
                "_type":type_,
                "_source":line
            }
            ACTIONS.append(action)
        success,_ = bulk(self.es,ACTIONS,index = index_, raise_on_error = True)
        print('Performed %d actions' % success)







if __name__ == "__main__":
    ST = StoreProse()
    #ST.getProse(r'E:\PycharmProject\FirstDayOnMS2\Data\Poem\processed_poem.json')
    # ST.getProse(os.path.join(r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem','processed_poem.json'))
    # ST.storeProse()


