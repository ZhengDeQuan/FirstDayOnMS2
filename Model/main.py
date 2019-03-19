import os

from Poem.crawlPoem import poemSanwenji , poemChinaSw
from Song.crawlSong import songSpider
from Song.crawlSong import MergeSongComment
from Poem_Song.PoemSongMatching import MatchPoemSong,MatchSeggedPoemSong
from Poem_Song.config import opt

from Poem.CleanData import PoemCleaner
from Poem.segPoem import SegmentationEngine,get_each_document,WriteBackToPoem

ProjectPath = "FirstDayOnMS2"
abspath = os.path.abspath(os.getcwd())
abspath = abspath.split(ProjectPath)
prefix_path = os.path.join(abspath[0],ProjectPath)

if __name__ == "__main__":

    # 散文的部分
    # spiderOnsanwenji= poemSanwenji(base_url=None)
    # spiderOnsanwenji.forward()
    # poemSpiderOnchinasw = poemChinaSw(base_url="http://www.sanwen.com/sanwen/jingdiansanwen/")
    # poemSpiderOnchinasw.forward()
    '''
    清洗散文
    '''
    # poemcleaner = PoemCleaner(
    #     [os.path.join(prefix_path,'Data\\Poem\\sanwenji\\sanwenji.json'),
    #      os.path.join(prefix_path,'Data\\Poem\\chinasw\\chinasw.json')],
    #     idf_path = "idf.txt")
    # poemcleaner.forward(os.path.join(prefix_path,'Data\\Poem\\processed_poem_2019.json'))
    '''
    切割散文
    '''
    #
    # Loading dataset
    #
    documents = get_each_document(os.path.join(prefix_path,'Data/Poem/processed_poem_2019.json'))
    engine = SegmentationEngine(n_topics=100, max_iter=70, a=0.1, b=0.01,
                                m=0.5)  # lda有两种训练方式，batch是默认的，更快，将所有数据导入内存训练；online，更慢，将数据分批导入内存训练
    print("the length of the documents = ", len(documents))
    X_train = documents
    X_test = documents
    # Input: SENTENCE
    print('SENTENCE')
    engine.fit(X_train, input_type='sentence')
    engine.pickle_lda("topicTilingWeights")
    engine.get_pickled_lda("topicTilingWeights")
    Res = engine.predict(X_test)
    WriteBackToPoem(os.path.join(prefix_path,'Data\\Poem\\processed_poem_2019.json'),Res,os.path.join(prefix_path,'Data\\Poem\\seged_poem_2019.json'))

    # 歌曲的部分
    # songClawer = songSpider()
    # songClawer.crawlSong()
    # songClawer = songSpider(target=['http://corechat-usermemory-int.trafficmanager.net:19200/'],
    #                         filter_by_comment=False, pklName=6)
    # songClawer.crawlSong(index='migu_music_merged_current')
    #merger = MergeSongComment(in_file=os.path.join(prefix_path,"Data\\Song\\song6.pkl"))
    # merger.MergeComment()
    #
    # 匹配的部分
    # poem_song_matcher = MatchSeggedPoemSong(opt=opt,
    #                                         poem_file=os.path.join(prefix_path,"Data\\Poem\\seged_poem_2019.json"),
    #                                         song_file=os.path.join(prefix_path,"Data\\Song\\song6.pkl"),
    #                                         out_file=os.path.join(prefix_path,'Data\\Poem_Song\\seggedProseSong.txt'),
    #                                         keywords=['夜晚', '深夜', '寂静', '安眠', '星空', '平静', '喧嚣', '静', '夜色', '月亮', "失眠"])
    # poem_song_matcher.forward()
