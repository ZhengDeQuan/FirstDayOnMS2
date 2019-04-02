# -*- coding:utf-8 -*-
'''
网上扒下来的unsupervised text segmentation算法,源码链接 https://github.com/ldulcic/text-segmentation/blob/master/TAR%20-%20Text%20Segmentation.ipynb
整体分为两个部分：
1.LDA（主题模型）
2.Text Tiling （Topic Tiling）
'''
from __future__ import division, print_function
import numpy as np
import nltk
from nltk.stem.porter import PorterStemmer
import string
import os
import jieba
import json
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.metrics.segmentation import pk
from nltk.metrics.segmentation import windowdiff
#关于pk和windowdiff这两个评分器的内容，详见：https://www.nltk.org/_modules/nltk/metrics/segmentation.html
from time import time
import pickle
import os
from scipy import spatial
from scipy.signal import argrelextrema
from Poem.config import seg_poem_opt

ProjectPath = "FirstDayOnMS2"
abspath = os.path.abspath(os.getcwd())
abspath = abspath.split(ProjectPath)
prefix_path = os.path.join(abspath[0], ProjectPath)

def load_chinese_stop_words(file = os.path.join(prefix_path,'Model\\stop_words\\stop_words.pkl')):
    return pickle.load(open(file, "rb"))
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
CHINESE_STOP_WORDS = load_chinese_stop_words()
CHINESE_PUNCTUATION = {'‘','’','“','”','（','）','《','》','！','？','；','、','：','。','，'}
stemmer = PorterStemmer()
# irrelevant characters specific for choi dataset
choi_noise = ['--', '\'\'', '``', ';', '..', '...', 'afj']


def is_digit(string):
    """
    Checks whether string is digit.
    :param string: String
    :return boolean: True if string is digit otherwise False.
    """
    return string.replace('.', '').replace('-', '').replace(',', '').isdigit()


def stem_tokens(tokens, stemmer):
    """
    Stemms tokens using Porter Stemmer.
    :param tokens: List of tokens.
    :param stemmer: Object which has method "stem" (in this poject it is nltk Porter Stemmer)
    :return list: List of stemmed tokens.
    """
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed


def tokenizeChinese(text,language = "Chinese"):
    """
    Tokenizes text.

    :param text: String.
    :return list: List of stemmed tokens from text.
    """
    if language == "Chinese":
        if type(text) == list:
            tokens = text
        else:
            tokens = list(jieba.cut(text))
        tokens = list(map(lambda t: t.lower(), tokens))
        tokens = filter(lambda t: t not in string.punctuation, tokens)  # filter ,条件为False的对象会被删除掉
        tokens = filter(lambda t: t not in CHINESE_PUNCTUATION,tokens)
        tokens = filter(lambda t: t not in CHINESE_STOP_WORDS, tokens)
        tokens = filter(lambda t: not is_digit(t), tokens)
        tokens = filter(lambda t: t not in choi_noise, tokens)
        return ' '.join(list(tokens))
    else:
        tokens = nltk.word_tokenize(text)
        tokens = list(map(lambda t: t.lower(), tokens))
        tokens = filter(lambda t: t not in string.punctuation, tokens) # filter ,条件为False的对象会被删除掉
        tokens = filter(lambda t: t not in ENGLISH_STOP_WORDS, tokens)
        tokens = filter(lambda t: not is_digit(t), tokens)
        tokens = filter(lambda t: t not in choi_noise, tokens)
        tokens = filter(lambda t: t[0] != '\'', tokens)  # remove strings like "'s"
        stems = stem_tokens(tokens, stemmer)
        return stems


def tokenize(text):
    """
    Tokenizes text.

    :param text: String.
    :return list: List of stemmed tokens from text.
    """
    tokens = nltk.word_tokenize(text)
    tokens = list(map(lambda t: t.lower(), tokens))
    tokens = filter(lambda t: t not in string.punctuation, tokens)
    tokens = filter(lambda t: t not in ENGLISH_STOP_WORDS, tokens)
    tokens = filter(lambda t: not is_digit(t), tokens)
    tokens = filter(lambda t: t not in choi_noise, tokens)
    tokens = filter(lambda t: t[0] != '\'', tokens)  # remove strings like "'s"
    stems = stem_tokens(tokens, stemmer)
    return stems


def get_filepaths(directory):#choi_data
    """
    Recursively searches directory and returns all file paths.

    :param directory: Root directory.
    :return list: List of all files in directory and all of its subdirectories.
    """
    file_paths = []

    for root, directories, files in os.walk(directory):

        for filename in files:
            # Join the two strings in order to form the full filepath.
            filepath = os.path.join(root, filename)
            file_paths.append(filepath)

    return file_paths


def doc_to_seg_string(n_sent, boundaries):
    """
    Creates string which represents documents (eg. '0000100001000')
    where 0 marks sentence and 1 marks boundary between segments.
    This string is used for evaluating topic tiling algorithm with Pk
    and WD measure.

    :param n_sent: Number of sentences in document.
    :param boundaries: Indices of boundaries between segments.
    :return string: String which represent document.
    """
    seg_string = ''
    for i in range(n_sent):
        if i in boundaries:
            seg_string += '1'
        else:
            seg_string += '0'
    return seg_string


def print_top_words(model, feature_names, n_top_words):
    """
    Prints top words for each topic where "model" is LDA model.
    """
    for topic_idx, topic in enumerate(model.components_):
        print("Topic #%d:" % topic_idx)
        print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]])) #取最后的三个，并且将其逆序了。比如[8,9,20]-->[20,9,8]
    print()


def max_left(sequence, ind_curr):
    """
    Searches for maximum value in sequence starting from ind_curr to the left.

    :param sequence: List of integer values.
    :param ind_curr: Index from which search starts.
    :return integer: Maximum value from in sequence where index is less than ind_curr.
    """
    max = sequence[ind_curr]
    while (ind_curr != 0) and (max <= sequence[ind_curr - 1]):
        max = sequence[ind_curr - 1]
        ind_curr -= 1
    return max


def max_right(sequence, ind_curr):
    """
    Searches for maximum value in sequence starting from ind_curr to the right.

    :param sequence: List of integer values.
    :param ind_curr: Index from which search starts.
    :return integer: Maximum value from in sequence where index is greater than ind_curr.
    """
    max = sequence[ind_curr]
    while (ind_curr != (len(sequence) - 1)) and (max <= sequence[ind_curr + 1]):
        max = sequence[ind_curr + 1]
        ind_curr += 1
    return max

class Document2:
    """
    Document represents one document.

    :param path: Path to file which contains document.
    :param sentences: list of document sentences
    :param boundaries: list of positions where segments boundaries are
                        (note: boundary 0 means segment boundary is behind sentence at index 0)
    :param segment_divider: string which indicates boundary between two segments
    """

    def __init__(self, path = None):
        """
        :param path: Path to file where documents is.
        """
        self.path = path
        self.sentences = []
        self.boundaries = []
        self.segment_divider = "=========="
        if path is not None:
            self.load_document(path)

    def load_document(self, path):
        """
        Loads document from file.

        :param path: Path to file where document is.
        """
        sentences = self.get_valid_sentences(path)
        for i, sentence in enumerate(sentences):
            print("sentence = ",sentence)
            if sentence != self.segment_divider: #读入的原始数据中，就在分割点处，有==========的字符串作为标志。
                self.sentences.append(sentence)
            else:
                self.boundaries.append(i - len(self.boundaries) - 1)
        # remove unecessary boundaries at beginning and the end
        del self.boundaries[0]
        del self.boundaries[-1] #因为原始数据中有

    def get_valid_sentences(self, path):
        """
        Reads all sentences from file and filters out invalid ones.
        Invalid sentences are sentences which are empty or contain
        only irrelevant tokens like stop words, punctuations, etc.

        :param path: Path to file where document is.
        :return list: List of valid sentences.
        """
        sentences = []
        with open(path, 'r') as fd:
            for line in fd:
                line = line.rstrip('\n')
                if tokenize(line):
                    # if line is valid
                    sentences.append(line)
        return sentences

    def to_text(self):
        """
        Returns this documents as appendend string of sentences.
        """
        return '\n'.join(self.sentences)

    def to_segments(self):
        """
        Returns this document as list of segments based on boundaries.
        """
        segments = []
        for i, boundary in enumerate(self.boundaries):
            if i == 0:
                segments.append(' '.join(self.sentences[0:boundary]))
            else:
                last_b = self.boundaries[i - 1]
                segments.append(' '.join(self.sentences[last_b:boundary]))
        segments.append(' '.join(self.sentences[self.boundaries[-1]:]))
        return segments

class Document:
    """
    Document represents one document.

    :param path: Path to file which contains document.
    :param sentences: list of document sentences
    :param boundaries: list of positions where segments boundaries are
                        (note: boundary 0 means segment boundary is behind sentence at index 0)
    :param segment_divider: string which indicates boundary between two segments
    """

    def __init__(self, path):
        """
        :param path: Path to file where documents is.
        """
        self.path = path
        self.sentences = []
        self.boundaries = []
        self.segment_divider = "=========="
        self.load_document(path)

    def load_document(self, path):
        """
        Loads document from file.

        :param path: Path to file where document is.
        """
        sentences = self.get_valid_sentences(path)
        for i, sentence in enumerate(sentences):
            # print("i = ",i)
            # print("sentence = ",sentence)
            if sentence != self.segment_divider: #读入的原始数据中，就在分割点处，有==========的字符串作为标志。
                self.sentences.append(sentence)
            else:
                self.boundaries.append(i - len(self.boundaries) - 1)
        # remove unecessary boundaries at beginning and the end
        del self.boundaries[0]
        del self.boundaries[-1] #因为原始数据中有

    def get_valid_sentences(self, path):
        """
        Reads all sentences from file and filters out invalid ones.
        Invalid sentences are sentences which are empty or contain
        only irrelevant tokens like stop words, punctuations, etc.

        :param path: Path to file where document is.
        :return list: List of valid sentences.
        """
        sentences = []
        with open(path, 'r') as fd:
            for line in fd:
                line = line.rstrip('\n')
                if tokenize(line):
                    # if line is valid
                    sentences.append(line)
        # print("path = ",path)
        # print(len(sentences))
        # print(sentences[0])
        # exit(89)
        return sentences

    def to_text(self):
        """
        Returns this documents as appendend string of sentences.
        """
        return '\n'.join(self.sentences)

    def to_segments(self):
        """
        Returns this document as list of segments based on boundaries.
        """
        segments = []
        for i, boundary in enumerate(self.boundaries):
            if i == 0:
                segments.append(' '.join(self.sentences[0:boundary]))
            else:
                last_b = self.boundaries[i - 1]
                segments.append(' '.join(self.sentences[last_b:boundary]))
        segments.append(' '.join(self.sentences[self.boundaries[-1]:]))
        return segments

class TopicTiling:
    """
    Implementation of Topic Tiling algorithm (M. Riedl and C. Biemann. 2012, Text Segmentation with Topic Models)

    :param m: Multiplier of standard deviation in condition for segmentation (c = mean - m*stddev).
    :param cosine_similarities: Cosine similarities between sentences.
    :param boundaries: Caluclated segments boundaries.

    """

    def __init__(self, m=0.5):
        self.m = m
        self.cosine_similarities = None
        self.boundaries = None

    def fit(self, sentence_vectors):
        """
        Runs Topic Tiling algorithm on list of sentence vectors.
        :param sentence_vectors: List (iterable) of topic distributions for each sentence in document.
                                 t-th element of sentence vector is weight for t-th topic for sentence,
                                 higher weight means that sentence is "more about" t-th topic.

        :return: Calculated boundaries based on similatities between sentence vectors.
                 Note: boundary '0' means that segment boundary is behind sentence which is at index 0
                 in sentence_vectors.
        """
        self.cosine_similarities = np.empty(0)
        depth_scores = []

        # calculating cosine similarities
        for i in range(0, len(sentence_vectors) - 1):
            sim = 1 - spatial.distance.cosine(sentence_vectors[i], sentence_vectors[i + 1]) # spatial.distance.cosine()计算的是1 - u . v / (|u| . |v|)
            self.cosine_similarities = np.append(self.cosine_similarities, sim)

        # get all local minima
        split_candidates_indices = argrelextrema(self.cosine_similarities, np.less_equal)[0] # 找出极小值点，<=的都算。比如：x = np.array([2, 1, 2, 3, 2, 0, 0, 0, 0, 1, 0]);  argrelextrema(x, np.less_equal); (array([ 1,  5,  6,  7,  8, 10], dtype=int64),)

        # calculating depth scores
        for i in split_candidates_indices:
            depth = 1 / 2 * (max_left(self.cosine_similarities, i) + max_right(self.cosine_similarities, i)
                     - 2 * self.cosine_similarities[i]) #深度 = （左边最大值 - 当前的局部最小 + 右边的最大值 - 当前的局部最小）/ 2
            depth_scores.append((depth, i)) #(当前点的深度，当前点)

        tmp = np.array(list(map(lambda d: d[0], depth_scores))) #tmp中只有每个点的深度

        # calculate segment threshold condition
        condition = tmp.mean() - self.m * tmp.std() # condition会作为是否分割的阈值，如果某个点的值小于等于， 均值 - self.m * 标准差，那么就切割这个点。

        # determine segment boundaries
        tmp = filter(lambda d: d[0] > condition, depth_scores)
        self.boundaries = [d[1] for d in tmp]

        return self.boundaries, depth_scores

    def set_m(self, m):
        """
        Setter for parameter m.
        """
        if m:
            self.m = m

class SegmentationEngine(BaseEstimator):
    """
    Implementation of segmentation engine used for segmenting documents.
    Based on Latent Dirichlet Allocation model and Topic Tiling algorithm.

    :param vectorizer: CountVectorizer class used for transforming and cleaning input data.
    :param lda: Latent Dirichlet Allocation model.
    :param tt: Topic Tiling class.
    :param n_topics: Number of topics parameter of LDA.
    :param max_iter: Maximum number of iterations parameter of LDA.
    :param a: Document topics prior parameter of LDA.
    :param b: Topic document prior parameter of LDA.
    :param m: Multiplier parameter of Topic Tiling
    :param random_state: Random state.
    """

    def __init__(self, n_topics=10, max_iter=None, a=None, b=None, m=None, random_state=None,lda_learning_method = "batch",opt = seg_poem_opt):
        """
        Initializes estimator.
        """
        self.n_topics = n_topics
        self.max_iter = max_iter
        self.a = a
        self.b = b
        self.m = m
        self.random_state = random_state

        self.vectorizer = CountVectorizer(max_df=0.95, min_df=2, tokenizer=tokenize, stop_words=CHINESE_STOP_WORDS)
        self.lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=max_iter, doc_topic_prior=a,
                                             topic_word_prior=b, random_state=random_state,learning_method=lda_learning_method)
        self.tt = TopicTiling(m=m)
        self.opt = opt

    def fit(self, documents, input_type='sentence'):
        """
        Trains segmentation engine.

        :param documents: List (iterable) of documents (class Document).
        :input_type: Determines basic input unit.Possible values are 'segment', 'document', 'sentence'.
                     By default we use 'segment'.
        """
        t0 = time()

        train_data = self.parse_data(documents, input_type)
        X = self.vectorizer.fit_transform(train_data)
        self.lda.fit(X)
        print('Fitted in %0.2f seconds' % (time() - t0))

    def pickle_lda(self,path):
        if not os.path.exists(path):
            os.mkdir(path)
        pickle.dump(self.vectorizer,open(os.path.join(path,'vectorizer.pkl'),"wb"))
        pickle.dump(self.lda,open(os.path.join(path,"lda.pkl"),"wb"))

    def get_pickled_lda(self , path):
        self.vectorizer = pickle.load(open(os.path.join(path,"vectorizer.pkl"),"rb"))
        self.lda = pickle.load(open(os.path.join(path,"lda.pkl"),"rb"))

    def predict(self, documents):
        """
        Calculates segment boundaries for documents.
        :param documents: List (iterable) of documents (class Document).
        :return: List of boundaries for each document.
        """
        # TODO check if fit has been called
        estimated_boundaries = []
        for document in documents:
            sentence_vectors = [self.lda.transform(self.vectorizer.transform([sentence])) for sentence in
                                document.sentences]
            '''sentence 是用空格间隔的 string type'''
            each_sentence_len = [len(''.join(sentence.split())) for sentence in document.sentences]
            boundaries, depth_scores = self.tt.fit(sentence_vectors)
            estimated_boundaries.append((boundaries, depth_scores, each_sentence_len))
        Res = self.infer_further(estimated_boundaries)
        return Res

    def infer_further(self, estimated_boundaries):
        Res = []
        def find_dp(start, end, each_sentence_len, depth_scores):
            '''
            在开区间(start,end)上寻找,使得分割点满足要求的，分割点，并且最好是分割在depth_score最大的点上
            算法，从depth_socre最大的开始增加，直到满足要求
            '''
            start_i = start + 1
            end_i = end - 1
            if start_i > end_i:
                return []
            seg_able_point = []
            for ele in depth_scores:
                if start_i<=ele[1]<=end_i:
                    seg_able_point.append(ele)
            if len(seg_able_point)==0:
                return []

            seg_able_point.sort(key = lambda t:t[0],reverse=True)

            for seg_num in range(1,len(seg_able_point)+1):
                Flag = True
                pre_bound = start-1
                for seg_point in seg_able_point[:seg_num]:
                    if not (self.opt['min_seg_length'] <= sum(each_sentence_len[pre_bound+1:seg_point[1]]) <= self.opt['max_seg_length']):
                        Flag = False
                        break
                    pre_bound = seg_point[1]
                if not (self.opt['min_seg_length'] <= sum(each_sentence_len[pre_bound + 1:end]) <= self.opt['max_seg_length']):
                    Flag = False
                if Flag:
                    return [ele[1] for ele in seg_able_point[:seg_num]]
            return []

        for (boundaries, depth_scores, each_sentence_len) in estimated_boundaries:
            print("boundaries = ",boundaries)
            print("depth_scores = ",depth_scores)
            print("each_sentence_len = ",each_sentence_len)
            '''
            sent =  秋季 随想
            sent =  秋天 多愁善感 夏季 华丽 落幕 中 悄然 而临 黄色 枯叶 空中 划出 一道 思索 弧线 悠然 沉寂 真 可谓 一叶知秋 相比 秋 洗刷 蔚蓝 高空 更 喜欢 渲染 火红 枫林 停车 坐爱 枫林晚 霜叶 红于 二月 花 枫林 秋 私语 述说 沧桑 生命 故事 秋雨 缠绵悱恻 世界 织 一条 精致 雨帘 可谓 大珠小珠落玉盘 清脆 声是 动听 催眠曲 秋 春 傲慢 夏 奔放 冬 冷酷 秋是 知性 感性 时常 思考 生命 意义 思索 生命 厚度 青春期 生命 索取 美丽 智慧 面对 挫折 只能 望天 兴叹
            sent =  纯真 童年 真挚 告白 不屑 笑容 留恋 踏入 青春期 面对 生命 变化 感到 无所适从 秋天 新 环境 新 老师 同学 新 学习 生活 生命 美丽 雾 置身其中 分辨 不清 雨 遮住 眼 虚无 寻找 奇迹 也许 生命 无尽 失望 中 找寻 一丝 希望 犹如 空谷回音 听 不到 真切 回答 听见 无助 呐喊 城市 灯塔 中喊出 愿望 总有一天 听见 城市 回复 城市 幸福 漂流瓶 命运 中 搁浅 找到 爱
            sent =  面对 生活 放荡不羁 谨小慎微 时常 听到 只缘身在此山中 无力 哀叹 青春 迷茫 生活 残酷 喜欢 沉浸 空想 中 海市蜃楼 美丽 是因为 神秘 世外桃源 令人 向往 是因为 束之高阁 清高 颐指气使 朋友 不解 中 幡然醒悟 清高 修身 处世 做 一支 出淤泥而不染 濯 清涟 妖 花朵 意志 前提 一朵 莲花 根 支撑 鲜艳 外表 只能 山花 烂漫 时 丛中 笑
            sent =  秋是 忧郁 多愁善感 依稀记得 红衣 女子 悲痛 余 葬花 故事 糟糕 成绩 中 潸然泪下 一只 蛹 尚未 摆脱 束缚 蓝色 蛹 等待 金色 碟 幻化 想 蛹 厚重 外壳 保护 安然 沉睡 受 不到 外界 风雨 侵袭 父母 温暖 双臂 做 美好 梦 蓝色 蛹 内心 充满 挣脱 外壳 悸动 外壳 破碎 声中 破茧 成蝶 金色 翅膀 金色 阳光 熠熠生辉 诧异 生 出手 奔跑 追逐 金色 蝶 金色 蝶 回头 渐渐 消失 眼帘 哭 泣不成声 是因为 孤独 是因为 懦弱 体会 温暖 港湾 停留 太久 世界 奋斗 拥有 生活 岁月蹉跎 黄了 树叶 绿 芭蕉 风雨兼程 中 刻骨 伤害 中 加深 生命 智慧 美丽 厚度
            sent =  擦干 眼泪 站 脚步 踉踉跄跄 一份 坚定 起书 主动 地去 汲取 生命 营养 一改 自命清高 常态 融入 朋友 主动 交谈 参加 活动 锻炼 面对 失败 挫折 依然 笑颜 如花 秋 忧伤 生命 传承 坠入 堕落 悬崖 众目睽睽 中 高姿态 面对 城市 星空 喊 出 时 早已 答案 答案 千磨 万击 坚劲 任尔 东西南北 风 一叶知秋 管中窥豹 努力 奋斗 中 生命 真谛 缓缓的 展现 面前 喜欢 深沉 热爱 生命
            boundaries =  [1]
            depth_scores =  [(0.4447948396505025, 1), (0.05610152935946028, 3)]
            each_sentence_len =  [4, 171, 137, 129, 204, 142]
            '''
            pre_bound_start = -1
            pre_bound = -1
            res = []
            boundaries.append(len(each_sentence_len)-1)
            Flag=True
            for indx, bound in enumerate( boundaries):
                if self.opt['min_seg_length'] <= sum(each_sentence_len[pre_bound+1:bound+1]) <= self.opt['max_seg_length']:
                    res.append(bound)
                    pre_bound_start = pre_bound + 1
                    pre_bound = bound
                elif sum(each_sentence_len[pre_bound+1:bound+1]) > self.opt['max_seg_length']:
                    temp = find_dp(pre_bound+1,bound,each_sentence_len,depth_scores) #在pre_bound+1 ，bound之间寻找一个满足长度要求的分割点，再次分隔
                    if temp:
                        res.append(pre_bound+1)
                        res.extend(temp)
                        res.append(bound)
                        pre_bound_start = pre_bound+1
                        pre_bound = bound
                    else:
                        Flag = False #这一段不存在满足条件的分割，只好放弃这个节目
                        break
                elif sum(each_sentence_len[pre_bound+1:bound+1]) < self.opt['min_seg_length']:
                    #如果遇到某一个段太短的情况的话，只能将这个段粘贴到前一段之后，或者下一段之前了
                    #注意当前的seg是开头的seg，和结尾的seg的情况。
                    '''
                    与之前的seg合并
                    '''
                    if indx != 0:
                        if len(res) >= 2:
                            if pre_bound_start != -1:
                                last_seg_start = pre_bound_start + 1
                            else:
                                last_seg_start = res[-2]
                                print("正常情况不会走这个分支")
                                assert 1==0
                        else:
                            last_seg_start = 0
                        if sum(each_sentence_len[last_seg_start:bound+1]) <= self.opt['max_seg_length']:
                            if len(res)>0:
                                res.pop(-1)
                            else:
                                res.append(0)
                                assert bound != 0
                            res.append(bound)
                            pre_bound_start = pre_bound + 1
                            pre_bound = bound
                            continue
                    '''
                    其余的情况都是要与下一个seg合并的
                    '''
                    if indx != len(boundaries)-1:
                        continue
                        #pre_bound不变，之间走到下一个bound
                    else:
                        #这是最后一个bound，肯定不可能拼出来了。
                        Flag = False
                        break
            if Flag:
                res.pop(0)
                Res.append(res)
            else:
                Res.append([])
        return Res


    def score(self, X, method='pk',k=None):
        """
        Calculates segmentation score with Pk or WindowDiff measure.

        :param X: List (iterable) of documents (class Document).
        :param method: String which indicates which evaluation method should be used.
                       Possible evaluation methods are Pk measure ('pk') or WindowDiff method ('wd').
                       By default Pk measure is used.
        :return float: Evaluation score (actually method returns 1 - pk or 1 - wd because standard
                       scikit learn grid search treats higher values as better while the oposite is
                       the case with pk and wd).
        """
        if method == 'wd':
            scorer = windowdiff
        else:
            scorer = pk

        scores = np.empty(0)
        estimated_boundaries = self.predict(X)
        for i, document in enumerate(X):
            ref_doc = doc_to_seg_string(len(document.sentences), document.boundaries)
            estimated_doc = doc_to_seg_string(len(document.sentences), estimated_boundaries[i])
            # calculate k
            if k is None:
                k = int(round(len(ref_doc) / (ref_doc.count('1') * 2.)))
            scores = np.append(scores, scorer(ref_doc, estimated_doc, k))
        return 1 - scores.mean()

    def set_params(self, **params):
        """
        Sets value of parameters.

        :param params: Dictionary of parameters to be set.
        """
        super(SegmentationEngine, self).set_params(**params)

        # refresh parameters
        self.lda.set_params(n_topics=self.n_topics, max_iter=self.max_iter, doc_topic_prior=self.a,
                            topic_word_prior=self.b, random_state=self.random_state)
        self.tt.set_m(self.m)
        return self

    def parse_data(self, documents, input_type='sentence'):
        """
        Transforms list of documents into list of segments.
        :param documents: List of documents (class Document)
        :input_type: Determines basic input unit.Possible values are 'segment', 'document', 'sentence'.
                     By default we use 'segment'.
        :return list: List of segments.
        """
        train_data = []
        for document in documents:
            if input_type == 'segment':
                train_data.extend(document.to_segments())
            elif input_type == 'sentence':
                train_data.extend(document.sentences)
            elif input_type == 'document':
                train_data.append(document.to_text())
            else:
                raise ValueError('Invalid input_type parameter!')
        return train_data

def get_each_document(path):
    documents=[]
    with open(path, "r", encoding='utf8') as fin:
        poems = json.load(fin)
        for one_poem in poems:
            paras = one_poem['paras']
            for one_para in paras:
                # one_para是一首散文，这个散文也可以有很多的段落，我将段落作为最小的切割单元，而不是句子。
                # para_content = one_para['para_content']
                fencied_para_content = one_para['fencied_para_content']
                if len(fencied_para_content) < 2: #就一个自然段就没有必要分了。
                    continue
                temp = Document2()
                for one in fencied_para_content:
                    #one 一个段落，一个list[word1 ,word2,.....]
                    one_paragraph = tokenizeChinese(one)
                    if one_paragraph:
                        temp.sentences.append(one_paragraph)
                documents.append(temp)
                # if len(documents) >= 2000 :#测试时为了快速
                #     return documents
    return documents

def WriteBackToPoem(path, Res , out_path):
    index = 0
    with open(path, "r", encoding='utf8') as fin:
        poems = json.load(fin)
        for poem_index , one_poem in enumerate( poems ):
            paras = one_poem['paras']
            seg_one_poem = dict()
            seg_one_poem['paras'] = []
            for para_index , one_para in enumerate( paras ):
                # one_para是一首散文，这个散文也可以有很多的段落，我将段落作为最小的切割单元，而不是句子。
                # para_content = one_para['para_content']
                seg_one_poem['paras'].append(one_para)
                fencied_para_content = one_para['fencied_para_content']
                if len(fencied_para_content) < 2:  # 就一个自然段就没有必要分了。
                    continue
                one_para['seg_point'] = list(map(int , Res[index])) #Res[index]中的数字都是np.int64的类型，不能用json序列化
                index += 1
                if index >= len(Res):
                    json.dump(poems, open(out_path, "w" , encoding="utf8"), ensure_ascii=False)
                    return
    return None

class SegPoem:
    def __init__(self,poem_file , out_file , opt = seg_poem_opt):
        self.poem_file = poem_file
        self.out_file = out_file
        self.opt = opt
        self.min_seg_length = opt['min_seg_length']
        self.max_seg_length = opt['max_seg_length']
        self.poems = json.load(open(poem_file,"r",encoding="utf-8"))


    def SegPoem(self):
        def SC(length):
            '''Satisfy length Constrain'''
            if self.min_seg_length<=length<=self.max_seg_length:
                return True
            return False

        def SegParaByRule(one_para):
            '''seg para by the length of each paragraph'''
            length = len(one_para['para_content'])
            if length == 1:
                if SC(len(one_para['para_content'][0])):
                    return [0]
                return []
            res = []
            seg_start = 0
            for seg_end in range(1,length):
                cur_len = len(''.join(one_para['para_content'][seg_start:seg_end]))
                if SC(cur_len):
                    res.append(seg_end-1)
                    seg_start = seg_end
                elif cur_len > self.max_seg_length:
                    return []
                elif cur_len < self.min_seg_length:
                    continue
            return res

        for one_poem in self.poems:
            for one_para in one_poem['paras']:
                rule_based_seg_points = SegParaByRule(one_para)
                print("rule_based_seg_points = ",rule_based_seg_points)
                one_para['rule_based_seg_points'] = rule_based_seg_points
        json.dump(self.poems,open(self.out_file,"w",encoding="utf-8"))



if __name__ == "__main__":

    documents = get_each_document(os.path.join(prefix_path,'Data\\Poem\\processed_poem_2019.json'))
    engine = SegmentationEngine(n_topics=100, max_iter=70, a=0.1, b=0.01, m=0.5)#lda有两种训练方式，batch是默认的，更快，将所有数据导入内存训练；online，更慢，将数据分批导入内存训练。
    print("the length of the documents = ",len(documents))
    X_train = documents
    X_test = documents
    # Input: SENTENCE
    print('SENTENCE')
    # engine.fit(X_train, input_type='sentence')
    # engine.pickle_lda(os.path.join(prefix_path,'Model\\topicTilingWeights'))
    engine.get_pickled_lda(os.path.join(prefix_path,'Model\\topicTilingWeights'))
    Res = engine.predict(X_test)
    WriteBackToPoem(os.path.join(prefix_path,'Data\\Poem\\processed_poem_2019.json'),Res,os.path.join(prefix_path,'Data\\Poem\\seged_poem_temp.json'))

