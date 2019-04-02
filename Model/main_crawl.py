import os
from Poem.crawlPoem import poemSanwenji , poemChinaSw
from Poem.PoemRawDataES import StoreRawProse
from config import opt

ProjectPath = "FirstDayOnMS2"
abspath = os.path.abspath(os.getcwd())
abspath = abspath.split(ProjectPath)
prefix_path = os.path.join(abspath[0],ProjectPath)

if __name__ == "__main__":

    # 散文的部分
    '''
    爬取散文
    '''
    # spiderOnsanwenji= poemSanwenji(base_url=None)
    # spiderOnsanwenji.forward()
    # poemSpiderOnchinasw = poemChinaSw(base_url="http://www.sanwen.com/sanwen/jingdiansanwen/")
    # poemSpiderOnchinasw.forward()
    '''
    es写散文
    '''
    SP = StoreRawProse()
    for path in opt['path_to_raw_poems']:
        SP.getProse(path)
        SP.storeProse()
        print("finish ",path)