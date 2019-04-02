import os
from Poem.crawlPoem import poemSanwenji , poemChinaSw
from Poem.PoemRawDataES import StoreRawProse
from config import path_opt

ProjectPath = "FirstDayOnMS2"
abspath = os.path.abspath(os.getcwd())
abspath = abspath.split(ProjectPath)
prefix_path = os.path.join(abspath[0],ProjectPath)

if __name__ == "__main__":

    # 散文的部分
    '''
    爬取散文
    '''
    # spiderOnsanwenji= poemSanwenji(base_url=None,
    #                                dir_to_save = path_opt['path_to_raw_poems'][0])
    # spiderOnsanwenji.forward()
    # poemSpiderOnchinasw = poemChinaSw(base_url="http://www.sanwen.com/sanwen/jingdiansanwen/",
    #                                   dir_to_save=path_opt['path_to_raw_poems'][1])
    # poemSpiderOnchinasw.forward()
    '''
    es写散文
    '''
    SP = StoreRawProse()
    for path in path_opt['path_to_raw_poems']:
        SP.getProse(path)
        SP.storeProse()
        print("finish ",path)