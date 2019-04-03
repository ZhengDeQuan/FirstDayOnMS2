import os
ProjectPath = "FirstDayOnMS2"
abspath = os.path.abspath(os.getcwd())
abspath = abspath.split(ProjectPath)
prefix_path = os.path.join(abspath[0],ProjectPath)

path_opt = {
    'path_to_raw_poems':[os.path.join(prefix_path,'Data\\Poem\\sanwenji\\sanwenji.json'),
                         os.path.join(prefix_path,'Data\\Poem\\chinasw\\chinasw.json')],
    'base_dir_for_save_raw_poem_from_es':os.path.join(prefix_path, "Data\\Poem"),
    'out_file_name_for_raw_poem_from_es':"poem_es.json",

    'idf_path':os.path.join(prefix_path,'Model', 'idf.txt'),
    'path_to_cleaned_poem':os.path.join(prefix_path,'Data\\Poem\\processed_poem_2019.json'),

    'path_to_segged_poem':os.path.join(prefix_path,'Data\\Poem\\seged_poem_2019.json'),

    'path_to_RULEsegged_poem':os.path.join(prefix_path,"Data\\Poem\\segedByRule_poem.json"),

    'base_dir_for_save_raw_song_from_es':os.path.join(prefix_path,'Data\\Song'),
    'out_file_name_for_raw_poem_from_es':"song.json",

    'base_dir_for_merge':os.path.join(prefix_path,'Data\\Song'),
    'out_file_name_for_merge_comment_song':"songComment.pkl",

    'path_for_match_result_viualize':os.path.join(prefix_path,'Data\\Poem_Song\\seggedProseSong.txt'),
    'path_for_match_result':os.path.join(prefix_path,"Data\\Poem_Song\\ForYaml"),

    'path_for_original_seed_word':os.path.join(prefix_path,"Model\\Poem\\seed_love.xlsx"),
    'path_for_save_selected_prose_by_seed_word':os.path.join(prefix_path, "Model\\Poem\\seed_love_story.xlsx"),

}

match_opt = {
    "key_words":['爱情'],
    'seg_point_field_name':'rule_based_seg_points',#选择以规则的方法切割的诗歌作为输入，'seg_points' is an alternative choice
    'additional_key_words_path':path_opt['path_for_original_seed_word'], #由Model/Poem/GetSeedWord.py对清洗后的散文，进行循环查找得到的种子词
}

