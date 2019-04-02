import os

from Poem.PoemRawDataES import GetRawProse
from Song.crawlSong import songSpider
from Song.crawlSong import MergeSongComment
from Poem_Song.PoemSongMatching import MatchPoemSong,MatchSeggedPoemSong,PoemMatchSong
from Poem_Song.config import opt

from Poem.CleanData import PoemCleaner
from Poem.segPoem import SegmentationEngine,get_each_document,WriteBackToPoem, SegPoem
from Poem.StorePoemToES import StoreProse
from Poem.GetPoemFromES import GetProse
ProjectPath = "FirstDayOnMS2"
abspath = os.path.abspath(os.getcwd())
abspath = abspath.split(ProjectPath)
prefix_path = os.path.join(abspath[0],ProjectPath)
from config import path_opt,match_opt

if __name__ == "__main__":

    # 散文的部分
    '''
    es读散文
    '''
    GRP = GetRawProse(base_save_dir = path_opt['base_dir_for_save_raw_poem_from_es'],
                 out_file_name = path_opt['out_file_name_for_raw_poem_from_es'])
    GRP.crawlPoem()
    '''
    清洗散文
    '''
    # poemcleaner = PoemCleaner(
    #     [os.path.join(path_opt['base_dir_for_save_raw_poem_from_es'],
    #                   path_opt['out_file_name_for_raw_poem_from_es'])],
    #     idf_path = path_opt['idf_path'])
    # poemcleaner.forward(dir_to_save = path_opt['path_to_cleaned_poem'])
    # '''
    # 切割散文
    # '''
    if match_opt["seg_point_field_name"] == "rule_based_seg_points":
        '''
        TopicTilling 切割
        '''
        documents = get_each_document(path = path_opt['path_to_cleaned_poem'])
        engine = SegmentationEngine(n_topics=100, max_iter=70, a=0.1, b=0.01,
                                    m=0.5)  # lda有两种训练方式，batch是默认的，更快，将所有数据导入内存训练；online，更慢，将数据分批导入内存训练
        print("the length of the documents = ", len(documents))
        X_train = documents
        X_test = documents
        # Input: SENTENCE
        engine.get_pickled_lda("topicTilingWeights")
        Res = engine.predict(X_test)
        WriteBackToPoem(path = path_opt['path_to_cleaned_poem'],Res = Res,out_path = path_opt['path_to_segged_poem'])
        '''
        存散文到es
        '''
        ST = StoreProse()
        ST.getProse(path_opt['path_to_segged_poem'])
        ST.storeProse(index_="prose_segged_by_algo", type_="prose")
    else:
        '''
        用规则的方法切割散文
        '''
        segger = SegPoem(poem_file=path_opt['path_to_cleaned_poem'],
                         out_file=path_opt['path_to_RULEsegged_poem'])
        segger.SegPoem()
        print("规则分割成功")
        '''
        存散文到es
        '''
        ST = StoreProse()
        ST.getProse(path_opt['path_to_RULEsegged_poem'])
        ST.storeProse(index_="prose_segged_by_rule", type_="prose")
    # '''
    # 从es上得到相应类别的散文
    # '''
    # # GP = GetProse(base_save_dir=os.path.join(prefix_path, "Data\\Poem"),out_file_name="seged_poem_20190402.json")
    # # GP.crawlPoem(index = 'prose_segged_by_algo' , key_words=['爱情','恋爱']) # alternative prose_segged_by_rule
    # # 歌曲的部分
    # '''
    # 从es上得到相应类别的歌曲
    # '''
    # # songClawer = songSpider()
    # # songClawer.crawlSong()
    # songClawer = songSpider(target=['http://corechat-usermemory-int.trafficmanager.net:19200/'],
    #                         filter_by_comment=False, pklName=0,
    #                         base_save_dir=path_opt['base_dir_for_save_raw_song_from_es'],
    #                         out_file_name=path_opt['out_file_name_for_raw_poem_from_es'],
    #                         )
    # songClawer.crawlSong(index='migu_music_merged_current')
    # # merger = MergeSongComment(in_file=os.path.join(path_opt['base_dir_for_save_raw_song_from_es'],path_opt['out_file_name_for_raw_poem_from_es']),
    # #                           out_file=path_opt['out_file_name_for_merge_comment_song'],
    # #                           base_dir=path_opt['base_dir_for_merge'])
    # # merger.MergeComment()
    #
    #
    # 匹配的部分
    # poem_song_matcher = MatchSeggedPoemSong(opt=opt,
    #                                         poem_file=os.path.join(prefix_path,"Data\\Poem\\seged_poem_20190402.json"),
    #                                         song_file=os.path.join(prefix_path,"Data\\Song\\song6.pkl"),
    #                                         out_file=os.path.join(prefix_path,'Data\\Poem_Song\\seggedProseSong.txt'),
    #                                         keywords=['夜晚', '深夜', '寂静', '安眠', '星空', '平静', '喧嚣', '静', '夜色', '月亮', "失眠"]
    #                                         )
    # poem_song_matcher.forward()
    if match_opt["seg_point_field_name"] == "rule_based_seg_points":
        poem_match_song = PoemMatchSong(opt=opt,
                                        poem_files=path_opt['path_to_RULEsegged_poem'],
                                        song_file=os.path.join(path_opt['base_dir_for_save_raw_song_from_es'],
                                                               path_opt['out_file_name_for_raw_poem_from_es']),
                                        keywords=match_opt['key_words'],
                                        seg_point_field_name=match_opt['seg_point_field_name'],  # seg_point
                                        out_file=path_opt['path_for_match_result_viualize'],
                                        idf_path=path_opt['idf_path'],
                                        additional_key_words_path=match_opt['additional_key_words_path'],
                                        out_txt_dir=path_opt['path_for_match_result'])
    else:
        poem_match_song = PoemMatchSong(opt=opt,
                                        poem_file=path_opt['path_to_segged_poem'],
                                        song_file=os.path.join(path_opt['base_dir_for_save_raw_song_from_es'],
                                                               path_opt['out_file_name_for_raw_poem_from_es']),
                                        keywords=match_opt['key_words'],
                                        seg_point_field_name = match_opt['seg_point_field_name'],#seg_point
                                        out_file=path_opt['path_for_match_result_viualize'],
                                        idf_path = path_opt['idf_path'],
                                        additional_key_words_path = match_opt['additional_key_words_path'],
                                        out_txt_dir=path_opt['path_for_match_result'])
    poem_match_song.forward()
    for turn_num in range(4,6):
        poem_match_song.opt['lower_bound']=turn_num
        poem_match_song.forward()
