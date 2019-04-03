
> Data
 
>> song\
   poem
   
> Model

>> model 
- Data 中存放数据和爬取数据的、预处理数据的逻辑
- Model 中存放匹配的逻辑,model中存放模型必要的权重

####
    运行所需额外数据说明：
    Model/idf.txt
    Model/stop_words/stop_words.pkl

####
    得到种子词的过程:
    Model/Poem/seed_love.xlsx 在这个文件的每一个sheet中的第一列上写入原始种子词，每个种子词一个单元格，从上到下
                              如果没有很多种子词，可以只新建一个sheet
    Model/Poem/GetSeedWord.py 运行这个文件。新抽取出的种子词，会存到seed_love.xlsx的新的sheet中。同时会生成新的文件seed_love_story.xlsx来存储被关键词选中的文章
    以上这两个.xlsx文件的位置，在Model/config.py 中的path_opt中的
    path_for_original_seed_word
    path_for_save_selected_prose_by_seed_word
    这两个key进行配置
####
    在Model/config.py path_opt中对程序的各种文件路径进行配置
    在Model/config.py match_opt中对匹配过程所使用的输入进行配置（包括主题关键词）
    在Model/Poem/config.py 中对清洗散文的要求参数比如最大最小长度等进行配置
    在Model/Poem_Song/config.py 中对文章与诗歌的匹配的阈值进行设定
    Model/Song/config 弃用