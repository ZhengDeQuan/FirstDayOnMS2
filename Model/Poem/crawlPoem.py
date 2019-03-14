import requests
from bs4 import BeautifulSoup
import json
import os
import re
from Model.utils.WebProxyTools import WebProxyTool

class poemSpider:
    def __init__(self,base_url,dir_to_save,base_dir_for_save = None, website_to_crawl = None):
        self.website_to_crawl = website_to_crawl
        self.base_url = base_url
        self.dir_to_save = dir_to_save
        self.base_dir_for_save = base_dir_for_save
        self.num_poems = 0
        self.poems = []
        self.url_crawled = []

    def getPage(self,href):
        req = requests.get(url=href)
        if req.status_code == 404:
            return None
        # req.encoding = "gb18030" # in case of Chinese character
        req.encoding = "gb18030"
        html = req.text
        bf = BeautifulSoup(html, 'html.parser')
        return bf

    def getOnePoem(self,href):
        html = self.getPage(href)
        texts = html.find(id="content")
        texts = texts.find(id="main")
        texts = texts.find_all('p')

        content = []
        for p in texts:
            content.append(p.get_text())
        content = content[2:]

        return content

    def getPoemList(self,target):
        bf = self.getPage(target)
        if bf is None:
            return None
        texts = bf.find_all('div', class_="list_content")[0]
        texts = texts.find_all('a')
        hrefs_to_crawl = []
        for a in texts:
            print(a.get_text())
            hrefs_to_crawl.append(a.get('href'))
        return hrefs_to_crawl

    def save(self,dir_to_save=None):
        if self.dir_to_save is None and dir_to_save is None:
            raise print(f'[log] no dir to save {self.website_to_crawl}')
        if dir_to_save is not None:
            with open(dir_to_save,'w',encoding="utf8") as fout:
                json.dump(self.poems,fout,ensure_ascii=False)
        if self.dir_to_save is not None and self.dir_to_save != dir_to_save:
            with open(self.dir_to_save,"w",encoding="utf8") as fout:
                json.dump(self.poems,fout,ensure_ascii=False)

class poemSanwenji(poemSpider):
    def __init__(self , base_url, dir_to_save=os.path.join(r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem','sanwenji/sanwenji.json'),base_dir_for_save = r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem',kind_url=None,base_name=None, website_to_crawl ='散文集网' ):
        super(poemSanwenji,self).__init__(base_url, dir_to_save, base_dir_for_save ,website_to_crawl = website_to_crawl)

    def getPage(self,href):
        req = requests.get(url=href)
        if req.status_code == 404:
            return None
        # req.encoding = "gb18030" # in case of Chinese character
        req.encoding = "gb18030"
        html = req.text
        bf = BeautifulSoup(html, 'html.parser')
        return bf

    def getOnePoem(self,href):
        html = self.getPage(href)
        texts = html.select('div[class="article_content"]')[
            0]  # 用select不要用find_all() ,后者返回的是一个NavigableString类型，而不是tag不能进行继续的检索
        texts = texts.select('p')
        content = []
        for p in texts:
            content.append(p.get_text())
        # content = content[2:]
        poem_title = []
        texts_article_title = html.select('div[class="article_tit"]')[0]
        texts_article_title = texts_article_title.select('a')[0].get_text()
        poem_title.append(texts_article_title)
        poem = poem_title + ['。'] + content
        return poem_title , content

    def getPoemList(self,target):
        bf = self.getPage(target)
        if bf is None:
            return None
        texts = bf.find_all('div', class_="list_content")[0]
        texts = texts.find_all('a')
        hrefs_to_crawl = []
        for a in texts:
            print(a.get_text())
            hrefs_to_crawl.append(a.get('href'))
        return hrefs_to_crawl

    def getOneKindPoem(self,base_url, save_dir_base,cur_poem_class):
        pre = base_url
        # pre = "http://www.sanwenji.cn/sanwen/sanwen/shanggan/list_"
        post = ".html"
        for number in range(1, 20):
            print("number = ", number)
            url = pre + str(number) + post
            hrefs_to_crawl = self.getPoemList(url)
            if hrefs_to_crawl is None:
                break
            for i, ele in enumerate(hrefs_to_crawl):
                print("number = ", number, " i= ", i)
                if ele in self.url_crawled:
                    continue
                self.url_crawled.append(ele)
                poem_title, poem_content = self.getOnePoem(ele)
                onePoem = poem_title + ['。'] + poem_content
                if len(onePoem) == 0:
                    print("ele = ", ele)
                    continue
                self.num_poems +=1
                temp_dict = {"poem_id":self.num_poems,
                             'poem_title':'\n'.join(poem_title),
                             'origin_poem':"\n".join(poem_content),
                             'poem_class':cur_poem_class,
                             'url':ele,
                             'website_name':self.website_to_crawl
                             }
                self.poems.append(temp_dict)

                '''
                下面是将这篇散文保存到一个文件的逻辑
                '''
                if self.base_dir_for_save:
                    save_dir_base = os.path.join(self.base_dir_for_save,save_dir_base)
                if not os.path.exists(save_dir_base):
                    os.makedirs(save_dir_base)
                print("save_dir_base",save_dir_base)
                filename = save_dir_base + str(number) + "__" + str(i)
                print("filename = ", filename)
                with open(filename + ".txt", "w", encoding="utf8") as fout:
                    fout.write("\n".join(onePoem))
                # with open(filename + '.pkl', "wb") as fout:
                #     pickle.dump(onePoem, fout)
                #     fout.close()

    def forward(self):
        self.kind_url = [
            "http://www.sanwenji.cn/sanwen/shuqing/list_",
            "http://www.sanwenji.cn/sanwen/sanwen/shanggan/list_",
            "http://www.sanwenji.cn/sanwen/youmeisanwen/list_",
            "http://www.sanwenji.cn/sanwen/suibi/list_",
            "http://www.sanwenji.cn/sanwen/xiandai/list_",
            "http://www.sanwenji.cn/sanwen/sanwen/xiejing/list_",
            "http://www.sanwenji.cn/sanwen/sanwen/lizhi/list_"
        ]
        self.base_name = 'sanwenji/'
        self.kind_name = []
        self.poem_classes = []
        for url in self.kind_url:
            cur_poem_class = url.split('/')[-2]
            self.poem_classes.append(cur_poem_class)
            url = self.base_name + url.split('/')[-2] + '/'
            self.kind_name.append(url)

        for u , n , c in zip(self.kind_url, self.kind_name , self.poem_classes):
            self.getOneKindPoem(u,n,c)

        self.save(self.dir_to_save)

class poemChinaSw(poemSpider):
    def __init__(self , base_url=None, dir_to_save=os.path.join(r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem','chinasw/chinasw.json'),base_dir_for_save = r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem',website_to_crawl = '中国散文网'):
        super(poemChinaSw,self).__init__(base_url, dir_to_save,base_dir_for_save,website_to_crawl = website_to_crawl)
        self.tool = WebProxyTool() #中国散文网有反扒机制
        self.crawled_list = []
        self.duplicate_num = 0

    def getPage(self,href):
        try:
            crawlable_url = self.tool.getCrawlUrl(href)
            req = requests.get(url=crawlable_url)
            if req.status_code == 404:
                return None
            # req.encoding = "gb18030" # in case of Chinese character
            req.encoding = "utf-8"
            html = req.text
            bf = BeautifulSoup(html, 'html.parser')
            return bf
        except:
            print("getPage Wrong")
            return None

    def getOnePoem(self,href):
        try:
            html = self.getPage(href)
            texts = html.select('div[class="row-article"]')[0]
            article = texts.select('h1')[0].get_text()
            content = texts.select('div[class="article-content"]')[0].get_text()
            content = content.replace("(中国散文网- www.sanwen.com)", "").split("中国散文网首发：http://www.sanwen.com")[-2]
            print(content)
            return article , content
        except:
            print(href)

    def getPoemList(self,target):
        bf = self.getPage(target)
        if bf is None:
            return None
        hrefs_to_crawl = []
        div = bf.select('div[class="list-base-article"]')[0]
        ul = div.select('ul')[0]
        lis = ul.select('li')
        for li in lis:
            a = li.select('a')[0]
            hrefs_to_crawl.append("http://www.sanwen.com" + a['href'])
        return hrefs_to_crawl

    def getOneKindPoem(self,base_url, save_dir_base,cur_poem_class):
        pre = base_url + "list_"
        post = ".html"
        # for number in range(20,100):
        for number in range(1, 20):
            print("number = ", number)
            url = pre + str(number) + post
            hrefs_to_crawl = self.getPoemList(url)
            if hrefs_to_crawl is None:
                break

            for i, ele in enumerate(hrefs_to_crawl):
                print("number = ", number, " i= ", i)
                if ele in self.crawled_list:
                    self.duplicate_num += 1
                    print("duplicat_num = ", self.duplicate_num)
                    continue
                else:
                    self.crawled_list.append(ele)
                poem_title, poem_content = self.getOnePoem(ele)
                onePoem = poem_title + "。" + poem_content
                if len(onePoem) == 0:
                    print("ele = ", ele)
                    continue
                self.num_poems += 1
                temp_dict = {"poem_id": self.num_poems,
                             'poem_title': poem_title,
                             'origin_poem': poem_content,
                             'poem_class': cur_poem_class,
                             'url': ele,
                             'website_name': self.website_to_crawl
                             }
                self.poems.append(temp_dict)
                '''
                下面是将这篇散文保存到一个文件夹的逻辑
                '''
                if self.base_dir_for_save:
                    save_dir_base = os.path.join(self.base_dir_for_save, save_dir_base)
                if not os.path.exists(save_dir_base):
                    os.makedirs(save_dir_base)
                print("save_dir_base", save_dir_base)
                filename = save_dir_base + str(number) + "__" + str(i)
                print("filename = ", filename)
                with open(filename + ".txt", "w", encoding="utf8") as fout:
                    fout.write("\n".join(onePoem))
                # with open(filename + '.pkl', "wb") as fout:
                #     pickle.dump(onePoem, fout)
                #     fout.close()

    def getKindList(self,url):
        try:
            html = self.getPage(url)
            hrefs = []
            divs = html.select('div[class="list-article-shanggan"]')[0]

            divs = divs.select('div[class="list-article-shanggan-box"]')

            for div in divs:
                li = div.select('li[class="head"]')[0]

                a = li.select('a')[0]
                hrefs.append("http://www.sanwen.com" + a['href'])
            return hrefs
        except:
            print("url = ", url)

    def forward(self):
        if self.base_url is None:
            self.base_url = "http://www.sanwen.com/sanwen/jingdiansanwen/"
        self.kind_url = self.getKindList(self.base_url)
        print("kind_url = ", self.kind_url)
        print("len = ", len(self.kind_url))
        self.poem_classes = [url.split('/')[-2] for url in self.kind_url]
        self.poem_classes = [re.sub('sanwen','',url) for url in self.poem_classes]
        print("poem_classes = ",self.poem_classes)
        self.base_name = 'chinasw/'
        self.kind_name = []
        for url in self.kind_url:
            url = self.base_name + url.split('/')[-2]+'/'
            print("url = ",url)
            self.kind_name.append(url)
        for u , n ,c in zip(self.kind_url,self.kind_name,self.poem_classes):
            try:
                self.getOneKindPoem(u,n,c)
            except:
                print("u = ",u)
                print("n = ",n)
        print("breaked")
        self.save()

class poemDusanwen(poemSpider):
    def __init__(self , base_url=None, dir_to_save=os.path.join(r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem','dusanwen/dusanwen.json'),base_dir_for_save = r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem',website_to_crawl = '文章阅读网'):
        super(poemDusanwen,self).__init__(base_url, dir_to_save,base_dir_for_save,website_to_crawl = website_to_crawl)
        self.tool = WebProxyTool() #中国散文网有反扒机制
        self.crawled_list = []
        self.duplicate_num = 0

    def makePoemObject(self, main_class, sub_class, href, page_title, Jokes):
        New_Jokes = []
        for joke in Jokes:
            joke = [re.sub(r'\(.*\)|（.*）', '', ele) for ele in joke]
            joke = [re.sub(r'^(0|1|2|3|4|5|6|7|8|9|0)',"",ele) for ele in joke]

            one_joke = {
                'main_class':main_class,'sub_class':sub_class,
                'url': href, 'page_title': page_title, "content": joke
            }
            New_Jokes.append(one_joke)
        return New_Jokes

if __name__ == "__main__":
    pass
    # spiderOnsanwenji= poemSanwenji(base_url=None)
    # spiderOnsanwenji.forward()
    # goon = input("go on? [Y/N]")
    # if not( goon.lower() == "y" or goon.lower() == "yes" ):
    #     exit(90)
    # poemSpiderOnchinasw = poemChinaSw(base_url="http://www.sanwen.com/sanwen/jingdiansanwen/")
    # poemSpiderOnchinasw.forward()

