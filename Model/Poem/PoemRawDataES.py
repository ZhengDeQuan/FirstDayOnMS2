from elasticsearch import Elasticsearch
from elasticsearch.helpers import bulk
import json
import os

ProjectPath = "FirstDayOnMS2"
abspath = os.path.abspath(os.getcwd())
abspath = abspath.split(ProjectPath)
prefix_path = os.path.join(abspath[0],ProjectPath)

class StoreRawProse:
    def __init__(self,target = ['http://chitchat-index-int.eastasia.cloudapp.azure.com:19200/'],
                 http_auth = ('esuser','Kibana123!'),
                 port = 9200 , timeout = 50000,
                 ):
        super(StoreRawProse,self).__init__()
        self.es = Elasticsearch(target, http_auth = http_auth, port = port, timeout = timeout)
        print("connect succeed!!!")
        self.prose = None

    def getProse(self , path):
        with open(path,"r",encoding="utf-8") as fin:
            self.prose = json.load(fin)

    def storeProse(self,index_ = "raw_prose",type_="raw_prose"):
        indices = self.es.cat.indices()
        indices = [ele['index'] for ele in indices]
        if index_ not in indices:
            result=self.es.indices.create(index=index_)
            print("result = ",result)
            print("over")
        # else:
        #     result=self.es.indices.delete(index=index_)
        #     print("result = ",result)
        if self.prose is None:
            raise("No raw prose provided!!!")
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


class GetRawProse:
    def __init__(self,target = ['http://chitchat-index-int.eastasia.cloudapp.azure.com:19200/'],
                 http_auth = ('esuser', 'Kibana123!'),
                 port=9200, timeout=50000,
                 base_save_dir = os.path.join(prefix_path, "Data\\Poem"),
                 out_file_name = "poem_es.json"
                 ):
        super(GetRawProse,self).__init__()
        self.target = target
        self.http_auth = http_auth
        self.port = port
        self.timeout = timeout
        self.es = Elasticsearch(target,http_auth= http_auth,port=port,timeout=timeout)
        self.base_save_dir = base_save_dir
        self.out_file_name = out_file_name
        print("connect succeed!!!")

    def crawlPoem(self,index = 'raw_prose/',doc_type="raw_prose"):
        body = {
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
            if len(self.ResultList) >= 300000: #为了安全，大于3十万的话需要分开存放了
                print("先保存下来")
                print("ResultNum = ", self.ResultNum)
                json.dump(self.ResultList,
                          open(os.path.join(self.base_save_dir, "poem" + str(self.pklName) + ".json"), "w",
                               encoding="utf-8"), ensure_ascii=False)
                self.ResultList = []
                self.ResultNum = 0
                self.pklName += 1
                print("save pkl succeed!")
                print("ResultNum = ", self.ResultNum)
                print("pklName = ", self.pklName)
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

