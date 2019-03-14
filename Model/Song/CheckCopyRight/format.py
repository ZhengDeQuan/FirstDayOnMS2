#coding:utf-8
"""
常用方法
Environment: python 3.5
"""
import re
import json
from Model.Song.CheckCopyRight import langconv as cov
from urllib.request import urlopen
from urllib.parse import quote


def zh2han(name):
    """
    中文繁简转换
    """
    for i in name:
        if u'\u4e00' <= i <= u'\u9fa6':
            name = cov.Converter('zh-hans').convert(name)
            break
    return name


def remove_brackets(name):
    """
    去除括号中的内容
    """
    name1 = re.sub(u'\{.*?\}|\(.*?\)|\[.*?\]|（.*?）|【.*?\】|［.*?］|｛.*?｝', '', name).strip()
    if name1 != '':
        name = name1
    return name


def remove_punctuation(name):
    """
    去除标点符号
    """
    name1 = re.sub(u'[\.\!\/\"\'\-·！，。？、+_,$%^*~+—~@#￥%……&*\[\]()《》<>（）『』「」｛｝【】{}]', '', name).strip()
    if name1 != '':
        name = name1
    return name


def remove_junks(name, junks=[' - live', ' - remix', 'instrumental$']):
    """
    移除不相关的词
    """
    name1 = re.sub(u'|'.join(junks), '', name).strip()
    if name1 != '':
        name = name1
    return name


def remove_multiblanks(name):
    """
    多个空格替换成1个空格
    """
    return ' '.join(name.split()).strip()


def onlinewb(message):
    """
    在线的分词
    """
    f=urlopen("http://wordbreaker.chinacloudapp.cn/api/breakword?query={0}&mkt=zh-cn&appid=0123456789".format(quote(message)))
    return json.loads(str(f.read(),encoding='utf-8'))["breakquery"]


def format_music(name):
    """
    处理歌曲名称
    """
    if name is None:
        return ''
    name = name.lower()
    name = zh2han(name)
    name = remove_junks(name)
    name = remove_brackets(name)
    name = remove_punctuation(name)
    name = remove_multiblanks(name)
    return name


def format_album(name):
    """
    处理专辑名称
    """
    if name is None:
        return ''
    name = name.lower()
    name = zh2han(name)
    name = remove_brackets(name)
    name = remove_punctuation(name)
    name = remove_multiblanks(name)
    return name


def format_singer(name):
    """
    处理歌手名称
    """
    if name is None:
        return ''
    name = name.lower()
    name = zh2han(name)
    name = remove_brackets(name)
    name = remove_punctuation(name)
    name = remove_multiblanks(name)
    return name
 

def extend_alias(name):
    """
    拓展别名
    """
    triggers = []
    triggers.append(' '.join(name.replace('\' ', '\'').split()))
    triggers.append(' '.join(name.replace('.', ' ').replace('-', ' ').replace(u'·', ' ').replace(u'\'', ' ').split()).strip())
    triggers.append(' '.join(name.replace('.', '').replace('-', '').replace(u'·', '').replace(u'\' ', '').replace(u'\'', '').split()).strip())
    res = set()
    res.add(name)
    for t in triggers:
        res.add(t)
    return res


"""
歌词相关的处理
"""
def extend(l):
    l1 = []
    for i in l:
        l1.append(i + u':')
        l1.append(i + u' : ')
        l1.append(i + u'：')
        l1.append(i + u' ： ')
        l1.append(i + ' ')
    return l1


written_composed_keys = extend([u'作词/作曲', u'作词 / 作曲', u'词 / 曲', u'词/曲', ])
written_keys = extend([u'作词', u'填词', u'词'])
composed_keys = extend([u'作曲' u'曲名', u'谱曲', u'原曲', u'曲',])
arranged_keys = extend([ u'编曲/混音', u'编曲', u'编曲',])
sung_keys = extend([u'演唱',  u'歌手名', u'表演', u'翻唱', u'歌手',])

def format_lyric(lyric):
    items = [i.strip() for i in lyric.split('\n')]
    written_by, composed_by, arranged_by, sung_by = None, None, None, None
    lyrics_with_time = {}
    lyrics = []
    for item in items:
        if len(item) > 0:
            # 繁简转换
            for i in item:
                if u'\u4e00' <= i <= u'\u9fa6':
                    item = zh2han(item)
                    break
            item = remove_multiblanks(item)

            
            flag = False
            # 作词作曲编曲演唱等信息抽取
            if item[:4] == '[ar:' and item[-1] == ']':
                #歌手名
                t = item[4:-1].strip()
                if len(t) > 0:
                    sung_by = t
                flag = True
            #if item[:4] == '[ti:' and item[-1] == ']':
                #曲名
            #if item[:4] == '[al:' and item[-1] == ']':
                #专辑名

            written_composed_key = [i for i in written_composed_keys if i in item]
            written_key = [i for i in written_keys if i in item]
            composed_key = [i for i in composed_keys if i in item]
            arranged_key = [i for i in arranged_keys if i in item]
            sung_key = [i for i in sung_keys if i in item]
            

            if len(written_composed_key) > 0:
                # 作词/作曲
                t = item.split(written_composed_key[0])[-1].strip()
                if len(t) > 0:
                    written_by = t if not written_by else written_by
                    composed_by = t if not composed_by else composed_by
                    #print ('{0}:{1}'.format('write/compose', t))
                    flag = True

            if len(written_key) > 0:
                # 作词
                t = item.split(written_key[0])[-1].strip()
                if len(t) > 0:
                    written_by = t  if not written_by else written_by
                    flag = True
                
                #print ('{0}:{1}'.format('write', t))
    
            if len(composed_key) > 0:
                # 作曲
                t = item.split(composed_key[0])[-1].strip()
                if len(t) > 0:
                    composed_by = t if not composed_by else composed_by
                    flag = True
                #print ('{0}:{1}'.format('compose', t))

            if len(arranged_key) > 0 :
                # 编曲
                t = item.split(arranged_key[0])[-1].strip()
                if len(t) > 0:
                    arranged_by = t if not arranged_by else arranged_by
                    flag = True
                #print ('{0}:{1}'.format('arrangede', t))
                
   
            if sung_by is None and len(sung_key) > 0:
                #表演
                t = item.split(sung_key[0])[-1].strip()
                if len(t) > 0:
                    sung_by = t if not sung_by else sung_by
                    flag = True
                #print ('{0}:{1}'.format('sung', t))

            for i in [u'制作：', u'后期：', u'歌名：', u'歌曲名：', u'歌名 ', u'歌曲名 ', u'歌名:', u'歌曲名:']:
                if i in item:
                    flag = True
                    break
            #歌词时间处理
            if not flag:                
                item = item.lower()    
                if item[0] == '[' and item[-1] == ']': 
                    continue
                valid_values = [value.strip() for value in re.split(r'\[|\]', item) if len(value.strip()) > 0]
                times, st = [], ''
                for value in valid_values:
                    contain_num = any(char.isdigit() for char in value)
                    contain_dot = any(char==':' or char=='：' or char==u'：'  for char in value)
                    contain_other = not (char==':' or char=='：' or char==u'：' or char.isdigit() for char in value)
                    if contain_num and contain_dot and not contain_other:
                        times.append(value)
                    elif not all('.' == char for char in value):
                        st += value + ' '
                
                if '====' in st or '--music--' in st or st in ['music ', u'（ music ) ', '<music> ', u'● ', u'→ ', 'music... ', 'end ', 'the end ', u'* * * '] or u'……………' in st or '----' in st or len(st.split('-')) == 2 or u'***' in st or '= m u i s e=' in st:
                    # 去除部分噪声
                    st = ''
                st = st.split(u'：')[-1]
                for i in (u'\(.*?\)',                    
                          u'（.*?）',
                          #u'<.*?>',
                          u'【.*?】'):
                    # 去除括号
                    st = re.sub(i, '', st)
                st = ' '.join(st.split()).strip()
                if st in [u'终', 'rap:']:
                    st = ''
                st = ' '.join(st.split())
                if len(st) > 2 and (st[1] == ':' or st[1] == ' ' or st[1] == u'：'):
                    st = st[2:]
                if len(st) > 0:
                    if len(times) > 0:
                        for time in times:
                            lyrics_with_time[time] = st
                    else:
                        lyrics.append(st) 
    return lyrics_with_time, lyrics, written_by, composed_by, arranged_by, sung_by


if __name__ == '__main__':
    # just test
    # print (format_music(u"久石譲1*^%#@!@^#*$(%)^*^^%^$%"))
    # print (extend_alias(u'羽·泉'))
    # print (onlinewb(u'羽·泉'))
    # print (format_music(u'ダンジョン オブ レガリアス主題歌'))
    # print (format_lyric('像没有眼睛的星星\n你亮着只闪烁你的寂寞\n像没有房间的温暖\n你空着只居住你的清冷\n像没有拥抱的问候\n你飞着只进入你的场景\n像没有泪水的呜咽\n你落下又回到你的哀伤\n任时光在荒老\n任岁月成蹉跎\n任热血被熬成了欲望\n任自己去原谅\n像没有眼睛的星星\n你亮着只闪烁你的寂寞\n像没有房间的温暖\n你空着只居住你的清冷\n任时光在荒老\n任岁月成蹉跎\n任热血被熬成了欲望\n任自己去原谅\n任时光在荒老\n任岁月成蹉跎\n任热血被熬成了欲望\n任自己去原谅\n任自己去原谅\n'))
    # res = format_lyric('像没有眼睛的星星\n你亮着只闪烁你的寂寞\n像没有房间的温暖\n你空着只居住你的清冷\n像没有拥抱的问候\n你飞着只进入你的场景\n像没有泪水的呜咽\n你落下又回到你的哀伤\n任时光在荒老\n任岁月成蹉跎\n任热血被熬成了欲望\n任自己去原谅\n像没有眼睛的星星\n你亮着只闪烁你的寂寞\n像没有房间的温暖\n你空着只居住你的清冷\n任时光在荒老\n任岁月成蹉跎\n任热血被熬成了欲望\n任自己去原谅\n任时光在荒老\n任岁月成蹉跎\n任热血被熬成了欲望\n任自己去原谅\n任自己去原谅\n')
    # lyrics_with_time, lyrics, written_by, composed_by, arranged_by, sung_by = res
    # print(lyrics_with_time)
    # print(lyrics)
    # print('\n'.join(lyrics))
    # print(written_by)
    # print(composed_by)
    # print(arranged_by)
    # print(sung_by)
    import pickle
    from tqdm import tqdm
    new_all_songs = []
    all_songs_with_keywords_copyright = pickle.load(open("../song/all_songs_with_keywords_copyright.pkl","rb"))
    for song in tqdm(all_songs_with_keywords_copyright):
        ori_lyric = song['lyric']['ori_lyric']
        lyrics_with_time, lyrics, written_by, composed_by, arranged_by, sung_by = format_lyric(ori_lyric)
        song['lyric']['formatted_lyric'] = '\n'.join(lyrics)
        new_all_songs.append(song)
    pickle.dump(new_all_songs,open('../song/all_song_with_keywords_copyright_formatLyric.pkl','wb'))