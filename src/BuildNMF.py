import numpy as np
import sklearn.decomposition
import matplotlib.pyplot as plt
import seaborn as sns
from get_tweets import pickle_results
from get_tweets import unpickle_results
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.porter import *
import sklearn.decomposition
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
import pdb
import numpy as np
from nltk.tokenize import TweetTokenizer
from collections import Counter, OrderedDict
import time


def load_tweet_corpus(fnames):
    '''
    When we have a quote tweet (tweet that is a response to a quoted tweet),
    combine the text of the quote tweet and the quoted tweet.

    fnames = list of pickle file names
    '''

    all_tweets = []
    for fname in fnames:
        all_tweets += unpickle_results(fname)

    tweet_text = [t.quote_or_rt_text + ' ' + t.all_text for t in all_tweets]
    tweet_corpus = np.array(list((set(tweet_text)))) # remove duplicates

    return tweet_corpus


class BuildNMF(object):
    """
    Description

    Methods
    --------


    Attributes
    -----------

    """

    def __init__(self, raw_tweet_corpus, num_topics):

        self.raw_tweet_corpus = raw_tweet_corpus
        self.num_topics = num_topics

        # Results from construction of TF-IDF matrix
        self.tf_idf = None
        self.stopwords = self._get_stopwords()
        self.words = None #list of words from TF-IDF
        self.vocab = None #dict mapping of TF-IDF words to indices
        self.top_words_in_topics = None #most frequent words in each NMF topic

        # Results from NMF (TF-IDF mat factored to produce W and H)
        self.W = None
        self.H = None
        self.reconstruction_err = None


    def fit(self, max_iter=100, solver='mu', topN=10, display=True):
        '''
        input: V matrix to be factorized by NMF, number of topics for NMF, max
        number of iterations for NMF factorization
        output: prints topN words in each topic if display=True
        returns: reconstruction error from NMF factorization
        '''

        self.fit_tf_idf()
        print('Now fitting NMF model (solver = ' + solver + ')')
        nmf = sklearn.decomposition.NMF(n_components=self.num_topics,
                                           max_iter=max_iter,
                                           solver=solver)

        self.W = nmf.fit_transform(self.tf_idf)
        self.reconstruction_err = nmf.reconstruction_err_
        self.H = nmf.components_

        self.top_words_in_topics = self._get_top_words_in_topics(topN=topN)

        if display:
            for idx, topic in enumerate(self.top_words_in_topics):
                print('-'*20)
                print('Most common words for Topic', idx, ':')
                for word in topic:
                    print(word)

        return nmf


    def fit_tf_idf(self):
        vec = TfidfVectorizer(strip_accents='ascii',tokenizer=self._tweet_tokenizer,
                              stop_words=self.stopwords, max_df=0.8, max_features=5000,
                              ngram_range=(1,1))

        self.tf_idf = vec.fit_transform(self.raw_tweet_corpus)
        self.words = np.array(vec.get_feature_names())
        self.vocab = vec.vocabulary_


    def get_umass_coherence_scores(self, M=5):
        scores = []
        for top_words_in_topic in self.top_words_in_topics:
            scores.append(self._get_umass_coherence_metric(top_words_in_topic,
                                                           M=5))

        return scores


    def _tweet_tokenizer(self, tweet_text, stem=True):
        tknzr = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)

        punctuation = string.punctuation.replace('#','')
        tokenized = tknzr.tokenize(tweet_text)
        tokenized = list(map(lambda x: x.strip(punctuation).replace("'",''),
                             tokenized))
        tokenized = list(filter(lambda x: x not in punctuation and len(x) > 2
                                and 'http' not in x, tokenized))

        if stem:
            stemmer = PorterStemmer()
            tokenized = [stemmer.stem(word) for word in tokenized]

        return tokenized


    def _get_stopwords(self):

        # NOTE: sklearn TfidfVectorizer removes stopwords AFTER tokenizing

        # get standard English stopwords and remove internal apostrophes
        english_stopwords = stopwords.words('english')
        english_stopwords = list(map(lambda x: x.replace("'",''), english_stopwords))

        # add stopwords
        additional_stopwords = ['changed', 'my', 'mind', 'view','opinion',
                                'fuck','lmao', 'shit', 'ok', 'okay', ' . ',
                                'lol']
        stemmer = PorterStemmer()
        additional_stopwords = [stemmer.stem(word) for word in additional_stopwords]
        all_stopwords =  english_stopwords + additional_stopwords

        return all_stopwords


    def _get_top_words_in_topics(self, topN=10):
        indices = np.array(np.flip(np.argsort(self.H, axis=1), axis=1))
        top_words_in_topics = []
        for row in indices:
            top_n_words = self.words[row][:topN]
            top_words_in_topics.append(top_n_words)

        return top_words_in_topics


    def _get_umass_coherence_metric(self, top_words_in_topic, M=10):
        terms_to_sum = []
        tf_idf = self.tf_idf
        tf_idf[tf_idf != 0] = 1

        for m in range(1, M):
            word_m = top_words_in_topic[m]
            word_m_index = self.vocab[word_m] # index of col corresponding to this
                                         # word in TF-IDF
            tfidf_column_for_word_m = tf_idf[:,word_m_index].todense()\
                                                            .astype(int)
            for l in range(0, m-1):
                word_l = top_words_in_topic[l]
                word_l_index = self.vocab[word_l]
                num_docs_with_word_l = tf_idf[:,word_l_index].sum()
                tfidf_column_for_word_l = tf_idf[:,word_l_index].todense()\
                                                                .astype(int)
                num_docs_with_both_words = (tfidf_column_for_word_m
                                            & tfidf_column_for_word_l).sum()
                result = (num_docs_with_both_words + 1)/num_docs_with_word_l
                terms_to_sum.append(result)

        coherence_score = np.sum(np.log(np.array(terms_to_sum)))

        return coherence_score


if __name__ == '__main__':
    raw_tweet_corpus = load_tweet_corpus(['../data/03_28_2018_18_02.pkl',
                                          '../data/03_30_2018_15_37.pkl'])

    nmf_mod = BuildNMF(raw_tweet_corpus, num_topics=15)
    nmf = nmf_mod.fit()