import json
import re
import os
import pickle
import jieba
from jieba import analyse
from typing import List , Dict ,AnyStr
from Poem.config import PoemClassPH, Festival, Season, XiaoiceCharacterSetting
from collections import Counter
ParaDict = Dict[str,str]
ParaList = List[ParaDict]
class PoemCleaner:
    def __init__(self,poemFiles=[],dir_to_save=os.path.join(r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem','processed_poem.json')):
        super(PoemCleaner,self).__init__()
        self.poem_id_offset = 0
        self.poems = []
        self.num_poem = 0
        self.dir_to_save = dir_to_save
        for poemfile in poemFiles:
            with open(poemfile,"r",encoding='utf8') as fin:
                poems = json.load(fin)
                if self.poem_id_offset != 0:
                    poems = self._get_offset_id(poems,self.poem_id_offset)
                self.poems.extend(poems)
                self.poem_id_offset = len(self.poems)
        print("length = ",len(self.poems))
        self.min_para_length = 200

    def _get_offset_id(self,poems,offset):
        if offset == 0:
            return poems
        new_poems = []
        for temp_dict in poems:
            temp_dict['poem_id'] += offset
            new_poems.append(temp_dict)
        return new_poems

    def WashText(self,origin_poem:str)->str:

        origin_poem = re.sub(r'\(中国散文网www\.SanWen\.com\)', '', origin_poem)
        origin_poem = re.sub(r'（中国散文网www\.SanWen\.com）', '', origin_poem)
        origin_poem = re.sub(r'\(中国散文网www\.sanwen\.com\)', '', origin_poem)
        origin_poem = re.sub(r'（中国散文网www\.sanwen\.com）', '', origin_poem)
        origin_poem = re.sub(r'\( 文章阅读网：www\.sanwen\.com \)', "", origin_poem)
        origin_poem = re.sub(r'（ 文章阅读网：www\.sanwen\.com ）', "", origin_poem)
        origin_poem = re.sub(r'\(中国散文网原创投稿 www\.sanwen\.com\)', "", origin_poem)
        # origin_poem = re.sub(r'\(中国散文网.*\)', '', origin_poem)
        # origin_poem = re.sub(r'\(.*\)', "", origin_poem)
        # origin_poem = re.sub(r'（.*）', "", origin_poem)
        origin_poem = re.sub(r"【作者：.*】", "", origin_poem)
        origin_poem = re.sub(r'文/未名人', '', origin_poem)
        origin_poem = re.sub(r'——题记', '', origin_poem)
        origin_poem = re.sub(r'-题记', '', origin_poem)
        origin_poem = re.sub(r'——', '', origin_poem)

        ele = [r'【一】', r'【二】', r'【三】', r'【四】', r'【五】', r'【六】', r'【七】', r'【八】', r'【九】', r'【后记】','原诗：']
        for e in ele:
            origin_poem = re.sub(e, "", origin_poem)
        def check_chinese(origin_poem):
            num_fb = 0
            for ch in origin_poem:
                if not ( u'\u4e00' <= ch <= u'\u9fff'):
                    num_fb += 1
            if num_fb > len(origin_poem) / 2:
                #print("origin_poem = ", origin_poem)
                return ""
            return origin_poem
        return check_chinese(origin_poem)

    def WashOff(self):
        '''
        洗掉原文中的网址等信息
        :return:
        '''
        new_poems = []
        for i , temp_dict in enumerate( self.poems ):
            origin_poem = temp_dict['origin_poem']

            origin_poem = self.WashText(origin_poem)
            if len(origin_poem) != 0:
                temp_dict['washed_poem'] = origin_poem
                poem_title = temp_dict["poem_title"]
                # if poem_title != "幸福的理由":
                #     continue
                new_poems.append(temp_dict)
        self.poems = new_poems

    def PoemSplit(self):
        '''
        origin_poem中的空白是很重要的信息，因为 这个是用来分隔原文的自然段的重要信息。
        split分两种，一种是分隔类似于篇一：，篇二：这种信息，每一个不同的篇章，都是一个可以独立作为散文的最小单元，这个由PoemSplitIntoSubPoeme完成
                    一种是分隔原文的自然段，这个对于summarize散文有用,这个由PoemSpiltIntoPassage完成
        以上两个信息之间有互指，所以要一起完成
        :return:
        '''
        for i , temp_dict in enumerate( self.poems ):
            washed_poem = temp_dict['washed_poem']
            washed_poem_paras = self.PoemSpiltIntoPassage(washed_poem)
            paras = self.PoemSplitIntoSubPoem(washed_poem_paras)
            self.poems[i]['paras'] = paras

    def PoemSpiltIntoPassage(self,washed_poem:str) -> List[str]:
        '''
        利用原文中的空白符分隔
        :return:list[str]
        '''
        return washed_poem.split()

    def PoemSplitIntoSubPoem(self,washed_poem_paras:List[str]) ->List[Dict[str,str]] :
        '''
        根据篇一、篇二这种天然的信息分隔
        list[dict('para_title':str,'para_content':str),....,dict()]
        :return:
        '''
        paras = []
        one_para = {'para_title':'','para_content':[]}
        for ele in washed_poem_paras:
            if ele.startswith(r"篇一：") or ele.startswith(r"篇二：") or ele.startswith(r"篇三：") \
            or ele.startswith(r"篇四：") or ele.startswith(r"篇五：") or ele.startswith(r"篇六：") \
            or ele.startswith(r"篇七：") or ele.startswith(r"篇八：") or ele.startswith(r"篇九：") \
            or ele.startswith(r"【篇一：") or ele.startswith(r"【篇二：") or ele.startswith(r"【篇三：") \
            or ele.startswith(r"【篇四：") or ele.startswith(r"【篇五：") or ele.startswith(r"【篇六：") \
            or ele.startswith(r"【篇七：") or ele.startswith(r"【篇八：") or ele.startswith(r"【篇九：") \
            or ele.startswith(r"【一】") or ele.startswith(r"【二】") or ele.startswith(r"【三】") \
            or ele.startswith(r"【四】") or ele.startswith(r"【五】") or ele.startswith(r"【六】") \
            or ele.startswith(r"【七】") or ele.startswith(r"【八】") or ele.startswith(r"【九】") \
            or ele.startswith(r"【后记】") \
            or ele.startswith(r"【"):
                # if flag:
                #     print("ele = ",ele)
                ele = re.sub(r"【篇一：", "", ele)
                ele = re.sub(r"【篇二：", "", ele)
                ele = re.sub(r"【篇三：", "", ele)
                ele = re.sub(r"【篇四：", "", ele)
                ele = re.sub(r"【篇五：", "", ele)
                ele = re.sub(r"【篇六：", "", ele)
                ele = re.sub(r"【篇七：", "", ele)
                ele = re.sub(r"【篇八：", "", ele)
                ele = re.sub(r"【篇九：", "", ele)
                ele = re.sub(r"篇一：", "", ele)
                ele = re.sub(r"篇二：", "", ele)
                ele = re.sub(r"篇三：", "", ele)
                ele = re.sub(r"篇四：", "", ele)
                ele = re.sub(r"篇五：", "", ele)
                ele = re.sub(r"篇六：", "", ele)
                ele = re.sub(r"篇七：", "", ele)
                ele = re.sub(r"篇八：", "", ele)
                ele = re.sub(r"篇九：", "", ele)
                ele = re.sub(r"【", "", ele)
                ele = re.sub(r"】", "", ele)

                one_para['para_title'] = ele
                paras.append(one_para)
                one_para = {'para_title':'','para_content':[]}
            else:
                one_para['para_content'].append(ele)
        paras.append(one_para)
        if paras[0]['para_title'] != "":
            try:
                assert paras[0]['para_content'] == []
            except:
                new_paras = []
                for one_para in paras:
                    if one_para['para_title']:
                        new_paras.append(one_para['para_title'])
                    if one_para['para_content']:
                        for ele in one_para['para_content']:
                            new_paras.append(ele)
                return [{'para_title':"","para_content":new_paras}]
        for i in range(len(paras) - 1):
            paras[i]['para_content'] = paras[i+1]['para_content']
        if len(paras) >=2:
            paras.pop(-1)
        return paras

    def RemoveSpecificChars(self, poem :str = "" ,chars_to_remove : List[str] = [r'【',r'】'])->str:
        for ele in chars_to_remove:
            poem=re.sub(ele,"",poem)
        return poem.strip()

    def PostWashOff(self):
        new_poems = []
        for i, temp_dict in enumerate(self.poems):
            new_para = []
            for para_dict in temp_dict['paras']:
                para_dict['para_title'] = self.RemoveSpecificChars(poem = para_dict['para_title'])
                inner_temp = []
                for inner_p in para_dict['para_content']:
                    inner_p = self.RemoveSpecificChars(poem = inner_p)
                    inner_temp.append(inner_p)
                if len(''.join(inner_temp)) < self.min_para_length: # poem中哪一首sub_poem比较短，就不要了
                    continue
                para_dict['para_content'] = inner_temp
                new_para.append(para_dict)
            temp_dict['paras'] = new_para
            if len(temp_dict['paras']) > 0:
                new_poems.append(temp_dict)
        self.poems = new_poems

    def ProcessPoemClass(self):
        '''
        爬取的时候散文的种类用拼音表示的，现在转换为汉字，转换的映射表在config.py中
        :return:
        '''
        poemClassTransfer = PoemClassPH()
        for i,poem_dict in enumerate( self.poems):
            website_name = poem_dict['website_name']
            poem_class = poem_dict['poem_class']
            index = poemClassTransfer.name_map[website_name]['pinyin'].index(poem_class)
            self.poems[i]['poem_class'] = poemClassTransfer.name_map[website_name]['hanzi'][index]

    '''
    以分词这个函数为界，之后的函数都是调用poem_dict['fencied_paras'] 不再调用poem_dict['paras']
    如果之后的函数调用时，没有看到这个字段，那么就调用self.Fenci()这个函数
    '''
    def Fenci(self):
        for i,poem_dict in enumerate( self.poems ):
            new_paras = []
            for para_dict in poem_dict['paras']:
                para_title = para_dict['para_title']
                if len(para_title.strip()) > 0:
                    para_dict['fencied_para_title'] = list(jieba.cut(para_title))
                else:
                    para_dict['fencied_para_title'] = []
                para_dict['fencied_para_content'] = []
                for para_c in para_dict['para_content']:
                    fencied_para_c = list(jieba.cut(para_c))
                    para_dict['fencied_para_content'].append(fencied_para_c)
                new_paras.append(para_dict)
            self.poems[i]['paras'] = new_paras

    def RemoveDuplicate(self):
        self.poem_string = []
        self.para_string = []
        new_poems = []
        new_poem_id = 0
        dup_poem = 0
        temp_dup_dict = {}
        for poem_dict in self.poems:
            poem_id = poem_dict["poem_id"]
            print("poem_id = ",poem_id)
            poem_content = []
            List_of_para_dict = []
            for para_dict in poem_dict['paras']:
                temp = [''.join(para_content) for para_content in para_dict['fencied_para_content']]
                temp = ''.join(temp)
                if temp not in self.para_string:
                    self.para_string.append(temp)
                    poem_content.append(temp)
                    List_of_para_dict.append(para_dict)
            poem_dict['paras'] = List_of_para_dict
            poem_content = ''.join(poem_content)
            if poem_content not in self.poem_string:
                temp_dup_dict[poem_content] = poem_dict
                self.poem_string.append(poem_content)
                poem_dict['poem_id'] = new_poem_id
                new_poem_id += 1
                new_poems.append(poem_dict)
            else:
                dup_poem += 1
                print("poem_content = ",poem_content)
                print("now    = ",poem_dict)
                print("before = ",temp_dup_dict[poem_content])
        print("dup_poem = ",dup_poem)

        self.poems = new_poems
        json.dump(self.para_string,open(os.path.join(r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem',"para_string.json"),"w",encoding="utf-8"),ensure_ascii=False)
        json.dump(self.poems,open(os.path.join(r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem',"REM_poem.json"),"w",encoding="utf-8"),ensure_ascii=False)

    def CheckTimingSig(self):
        '''
        check whether the topic of a poem belongs to a festival or is a season
        TimingSig refers to festival or season
        :return:
        '''
        f = Festival()
        s = Season()
        def checkByPara(paras,festival_list,season_list):
            '''
            :param paras:
            :param festival_list:
            :param season_list:
            :return:
            '''
            season , festival = [] , []
            for i,para_dict in enumerate( paras ) :
                paras[i]['season'] = []
                paras[i]['festival'] = []
                para_title = para_dict['fencied_para_title']
                if len(para_title) > 0:
                    for word in para_title:
                        if word in season_list:
                            season.append(word)
                            paras[i]['season'].append(word)
                        if word in festival_list:
                            festival.append(word)
                            paras[i]['festival'].append(word)

                for para_c in para_dict['fencied_para_content']:

                    for word in para_c:
                        if word in season_list:
                            season.append(word)
                            paras[i]['season'].append(word)
                        if word in festival_list:
                            festival.append(word)
                            paras[i]['festival'].append(word)
                print(paras[i]['season'])
                print(paras[i]['festival'])

            return season, festival , paras

        for i , poem_dict in enumerate( self.poems ):
            poem_class = poem_dict['poem_class']
            if poem_class in f.festival_list:
                if "festival" not in poem_dict:
                    poem_dict['festival'] = [poem_class]
                else:
                    poem_dict['festival'].append(poem_class)

            if poem_class in s.season_list:
                if 'season' not in poem_dict:
                    poem_dict['season'] = [poem_class]
                else:
                    poem_dict['season'].append(poem_class)

            season, festival , new_paras = checkByPara(poem_dict['paras'],f.festival_list,s.season_list)
            poem_dict['paras'] = new_paras
            if 'season' not in poem_dict:
                poem_dict['season'] = season
            else:
                poem_dict['season'].extend(season)
            if 'festival' not in poem_dict:
                poem_dict['festival'] = festival
            else:
                poem_dict['festival'].extend(festival)
            self.poems[i] = poem_dict

    def CheckCharacter(self):
        '''
        检查不符合小冰人设的散文，小冰的人设是18岁少女
        :return:
        '''
        forbidden = XiaoiceCharacterSetting.forbidden_keywords
        for poem_dict in self.poems:
            new_paras = []
            for para_dict in poem_dict['paras']:
                flag_has_forbidden = False
                for para_content in para_dict['fencied_para_content']:
                    for word in para_content:
                        if word in forbidden:
                            flag_has_forbidden = True
                            break
                    if flag_has_forbidden:
                        break
                if not flag_has_forbidden:
                    new_paras.append(para_dict)
            poem_dict['paras'] = new_paras

    def ExtractKeyWord(self):
        for i , poem_dict in enumerate(self.poems):
            new_paras = []
            for j,para_dict in enumerate(poem_dict['paras']):
                all_content = '\n'.join(para_dict['para_content'])
                key_words = analyse.textrank(all_content, topK=10, withWeight=True, allowPOS=('ns', 'n', 'vn'))#'v' #ns地名；n名词；vn名动词，比如思索；v动词
                para_dict['key_words'] = key_words
                new_paras.append(para_dict)
            self.poems[i]['paras'] = new_paras

    def save(self,dir_to_save = None):
        if self.dir_to_save is None and dir_to_save is None:
            raise ValueError("no dir to save")
        if dir_to_save is not None:
            with open(dir_to_save,"w",encoding="utf8") as fout:
                json.dump(self.poems,fout,ensure_ascii=False)
        else:
            with open(self.dir_to_save,"w",encoding="utf8") as fout:
                json.dump(self.poems,fout,ensure_ascii=False)

    def load(self,dir_to_load = None):
        if dir_to_load is None and self.dir_to_save is None:
            raise ValueError("no dir to load")

        if dir_to_load is not None and os.path.exists(dir_to_load):
            with open(dir_to_load,"r",encoding="utf8") as fin:
                self.poems = json.load(fin)
        elif self.dir_to_save is not None and os.path.exists(self.dir_to_save):
            with open(self.dir_to_save,"r",encoding="utf8") as fin:
                self.poems = json.load(fin)
        else:
            raise ValueError("load dir not exists")

    def CountZero(self):
        num = 0
        for poem_dict in self.poems:
            if len(poem_dict['paras']) == 0:
                print(poem_dict)
                num+=1
        print("num = ",num)

    def forward(self,dir_to_save = None):
        print("1 = ",len(self.poems))
        self.WashOff()
        print("washOff = ",len(self.poems))

        self.PoemSplit()
        print("split = ",len(self.poems))

        self.PostWashOff()
        print("postWash = ",len(self.poems))


        self.ProcessPoemClass()
        print("PoemClass = ",len(self.poems))

        self.CountZero()

        self.Fenci()
        print("Fenci = ",len(self.poems))

        self.RemoveDuplicate()
        print("RD = ",len(self.poems))

        self.CheckTimingSig()
        print("CT = ",len(self.poems))

        # self.CheckCharacter() #不再需要过滤人设
        self.ExtractKeyWord()
        print("EK = ",len(self.poems))

        self.save(dir_to_save+".json")






if __name__ == "__main__":
    poemcleaner = PoemCleaner([os.path.join(r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem','sanwenji/sanwenji.json'),os.path.join(r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem','chinasw/chinasw.json')])
    poemcleaner.forward(os.path.join(r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem','processed_poem_2019.json'))
    # poemcleaner = PoemCleaner(
    #     [os.path.join(r'E:\\PycharmProjects\\FirstDayOnMS2\\Data\\Poem', 'sanwenji/sanwenji.json')]
    # )
    # poemcleaner.forward()