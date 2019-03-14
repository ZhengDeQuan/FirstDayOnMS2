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

Mode = "Test"
"""
1.确实输入数据格式
"""

#
#Helper functions
#
import pickle
import os
from scipy import spatial
from scipy.signal import argrelextrema



def load_chinese_stop_words(file = 'stop_words/stop_words.pkl'):
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

#
#Document class
#

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


#
#Topic Tiling class
#


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

        return self.boundaries

    def set_m(self, m):
        """
        Setter for parameter m.
        """
        if m:
            self.m = m

#
#Segmentation Engine class
#

from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.metrics.segmentation import pk
from nltk.metrics.segmentation import windowdiff
#关于pk和windowdiff这两个评分器的内容，详见：https://www.nltk.org/_modules/nltk/metrics/segmentation.html
from time import time


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

    def __init__(self, n_topics=10, max_iter=None, a=None, b=None, m=None, random_state=None,lda_learning_method = "batch"):
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
            estimated_boundaries.append(self.tt.fit(sentence_vectors))
        return estimated_boundaries

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
                    #one 一个段落，一个list[word1 ,word2]
                    one_paragraph = tokenizeChinese(one)
                    if one_paragraph:
                        temp.sentences.append(one_paragraph)
                documents.append(temp)
                if len(documents) >= 2000 :#测试时为了快速
                    return documents
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

if __name__ == "__main__":
    #
    # Loading dataset
    #
    documents = get_each_document('poem_data/processed_poem.json')
    #engine = SegmentationEngine(n_topics=100, max_iter=70, a=0.1, b=0.01, m=0.5 , lda_learning_method="online")
    engine = SegmentationEngine(n_topics=100, max_iter=70, a=0.1, b=0.01, m=0.5)#lda有两种训练方式，batch是默认的，更快，将所有数据导入内存训练；online，更慢，将数据分批导入内存训练。
    # splitter = ShuffleSplit(len(documents), n_iter=1, test_size=.05, random_state=273)
    # X_train = []
    # X_test = []
    # for train_indices, test_indices in splitter:
    #     X_train = [documents[i] for i in train_indices[:100]]  # take 100 documents , 只取100个应该是为了验证程序的有效性，只取少量数据
    #     X_test = [documents[i] for i in test_indices]

    print("the length of the documents = ",len(documents))
    X_train = documents
    X_test = documents
    # Input: SENTENCE
    print('SENTENCE')
    engine.fit(X_train, input_type='sentence')
    engine.pickle_lda("topicTilingWeightsChoi")
    engine.get_pickled_lda("topicTilingWeightsChoi")
    Res = engine.predict(X_test)
    # WriteBackToPoem('poem_data/processed_poem.json',Res,'poem_data/seged_poem.json')


    # print('Pk = %f' % (1 - engine.score(X_test)))
    # print('WD = %f' % (1 - engine.score(X_test, method='wd')))
    #
    # '''
    # SEGMENT
    # Fitted in 91.38 seconds
    # Pk = 0.175918
    # WD = 0.223952
    # DOCUMENT
    # Fitted in 44.14 seconds
    # Pk = 0.360908
    # WD = 0.411459
    # SENTENCE
    # Fitted in 216.62 seconds
    # Pk = 0.459176
    # WD = 0.533597
    # '''
    #
    # #
    # # Train-test split
    # #
    #
    # X_train, X_test = train_test_split(documents, test_size=0.2, random_state=273)
    #
    # #
    # #
    # # Grid Search
    # # Optimizing parameters $K$, $\alpha$ and $x$
    # # WARNING! Very slow block DO NOT RUN THIS!!!
    # #
    # #
    #
    # # from sklearn.grid_search import GridSearchCV
    # #
    # # cv = ShuffleSplit(len(X_train), n_iter=1, test_size=.2)
    # #
    # # engine = SegmentationEngine(max_iter=60, a=0.1, b=0.01)
    # # params = {'n_topics': [60, 80, 100, 120, 130, 140, 150, 160], 'm': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
    # #           'a': [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1]}
    # # clf = GridSearchCV(engine, params, cv=cv)
    # #
    # # t = time()
    # # clf.fit(X_train)
    # # print('Duration %f hours' % ((time() - t) / 3600))
    #
    # # Model with optimal parameters
    # engine = SegmentationEngine(n_topics=150, max_iter=100, a=0.1, b=0.01, m=0.2, random_state=273)
    # engine.fit(X_train)
    # #Fitted in 1587.52 seconds
    #
    # # score function is returning(1 - score) because scikit - learn grid search treats higher value as better(which is opposite for Pk and WD)
    #
    # print('Pk = %f' % (1 - engine.score(X_test, method='pk')))
    # print('WD = %f' % (1 - engine.score(X_test, method='wd')))
    #
    # # Pk = 0.096507
    # # WD = 0.126073
    #
    # #Choose which document to plot and generate plot indices
    # doc = X_test[174]  # just change index for chosing different document
    # plot_indices = engine.predict([doc])[0]

