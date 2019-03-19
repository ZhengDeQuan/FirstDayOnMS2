opt = {
    'poem_threshold':0.1,
}


class PoemClassPH:
    def __init__(self):
        # 散文的类型，爬取的时候散文的种类用拼音表示的，拼音转汉字
        self.chinasw = {
            'pinyin':['aiqing', 'chunjie', 'dongtian', 'gudai', 'guoqingjie', 'guxiang', 'jiaoshijie', 'jieri', 'jingmei',
        'jishi', 'meng', 'mingjia', 'muai', 'qingchun', 'qinggangushi', 'qinggan', 'qingmingjie', 'qingrenjie',
         'qinqing', 'qiutian', 'qixi', 'shi', 'shanggan', 'shengdanjie', 'shuqing', 'xiatian', 'xiejing', 'xieren',
         'xiewu', 'xue', 'youmei', 'youqing', 'yuanxiaojie', 'zheli'],

            'hanzi':['爱情','春节','冬天','古代','国庆节','故乡','教师节','节日','精美',
        '记事','梦','名家','母爱','青春','情感故事','情感','清明节','情人节',
         '亲情','秋天','七夕','诗','伤感','圣诞节','抒情','夏天','写景','写人',
         '写物','雪','优美','友情','元宵节','哲理']
        }

        self.sanwenji = {
            'pinyin':['lizhi','shanggan', 'shuqing', 'suibi', 'xiandai', 'xiejing', 'youmeisanwen'],
            'hanzi':['励志','伤感','抒情','随笔','现代','写景','优美']
        }
        self.name_map = {
            '散文集网':self.sanwenji,
            '中国散文网':self.chinasw
        }


class Festival:
    def __init__(self):
        self.festival_list = ['中秋', '元宵', '端午', '元旦', '春节', '除夕', '新年',
                     '圣诞', '平安夜',
                     '清明节', '重阳', '寒食节',
                     '国庆', '儿童节', '劳动节', '青年节','教师节'
                     '感恩节', '万圣节', '愚人节',
                     '情人节', '七夕'
                     ]

class Season:
    def __init__(self):
        self.season_list = [
            '春天','夏天','秋天','冬天'
        ]


class XiaoiceCharacterSetting:
    forbidden_keywords = [
        '女儿', '儿子', '侄子', '侄女',
        '丈夫', '妻子',
        '结婚', '离婚', '离异',
    ]
    def __init__(self):
        '''
        小冰的人物设定是18岁的少女，所以可以有妈妈，爸爸，但是不能有女儿，儿子，这些不能有的情况，通过关键词来过滤
        '''
        # self.forbidden_keywords = [
        #     '女儿','儿子','侄子','侄女',
        #     '丈夫','妻子',
        #     '结婚','离婚','离异',
        # ]
if __name__ == "__main__":
    A = PoemClassPH()

