import numpy as np
import sklearn.decomposition
import matplotlib.pyplot as plt
import seaborn as sns
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
from get_tweet_corpus import TweetCorpus
import time


class BuildNMF(object):
    """
    Given a corpus of tweets, this class creates a TF-IDF matrix and builds an
    NMF model in order to find latent topics in the tweet corpus. This class
    also allows the user to compute UMass coherence scores for each topic in
    order to evaluate the coherence of the latent topics.

    Methods
    --------
    - fit():
        Uses NMF to find latent topics in tweet corpus and prints the most
        common words in each topic in decreasing order of frequency and returns
        the fitted scikit-learn NMF object
    - _fit_tf_idf():
        helper function that constructs TF-IDF matrix from tweet corpus and
        saves the list of words and a dict mapping from words to TF-IDF indices
    - get_umass_coherence_scores():
        computes coherence scores for all latent topics and returns the scores
        in the form of a list
    - _tweet_tokenizer():
        helper function for _fit_tf_idf() that cleans (removes punctuation and
        stopwords) and tokenizes tweets; also creates reverse lookup table
        so stemmed words can later be mapped to full words
    - _get_stopwords():
        helper function to get list of stopwords (standard english stopwords
        included in nltk.corpus library plus a few added stopwords)
    - _get_top_words_in_topics():
        helper function to get the most commonly occuring words in each topic
    - _get_umass_coherence_metric():
        helper function for get_umass_coherence_scores() that simply computes
        the UMass coherence score for a single topic
    - get_tweets_in_each_topic():
        build a dict mapping topic indices (keys) to the corresponding
        tweets in that topic (values)


    Attributes
    -----------
    - tweet_corpus (np array):
        array of strings (each string is a tweet)
    - num_topics (int):
        number of latent topics
    - tf_idf (np array of floats):
        TF-IDF matrix created from tweet_corpus
    - stopwords (list of strings):
        list of words ignored when constructing TF-IDF
    - words (np array of strings):
        list of all (stemmed/lowercased) words in the construction of the
        TF-IDF matrix
    - vocab (dict):
        mapping of words (strings) to column indices (ints) in TF-IDF matrix
    - top_words_in_topics (list of lists):
        lists of top N words (strings) in each latent topic (e.g., 0th index
        of top_words_in_topics = list of topN words in 0th topic)
    - W (np array):
        W matrix from non-negative matrix factorization of TF-IDF
    - H (np array):
        H matrix from non-negative matrix factorization of TF-IDF
    - reconstruction_err (float):
        Frobenius norm of TF-IDF - WH

    """

    def __init__(self, tweet_corpus, num_topics):
        self.tweet_corpus = tweet_corpus
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

        self.stem_lookup_table = {} # dict mapping of stems to original words


    def fit(self, max_iter=100, solver='mu', topN=10, display=True):
        """
        Uses NMF to find latent topics in tweet corpus and prints the most
        common words in each topic in decreasing order of frequency

        inputs:
        - max_iter (int): max number of iterations in sklearn NMF computation
        - solver (string):
            -'cd' = coordinate descent (gradient descent method for minimizing
                    reconstruction error in NMF)
            - 'mu' = multiplicative update
        - topN (int): number of most common words in each topic to get
        - display (boolean): if True then print the topN words in each topic

        returns:
        - fitted scikit-learn NMF object
        """

        self._fit_tf_idf()
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
                print('Most common (stemmed) words for Topic ' + str(idx) + ':')
                for word in topic:
                    print(word)

        return nmf


    def _fit_tf_idf(self):
        """
        helper function that constructs TF-IDF matrix from tweet corpus and
        saves the list of words and a dict mapping from words to TF-IDF indices
        """

        vec = TfidfVectorizer(strip_accents='ascii',tokenizer=self._tweet_tokenizer,
                              stop_words=self.stopwords, max_df=0.8, max_features=5000,
                              ngram_range=(1,1))

        self.tf_idf = vec.fit_transform(self.tweet_corpus)
        self.words = np.array(vec.get_feature_names())
        self.vocab = vec.vocabulary_


    def get_umass_coherence_scores(self, M=5):
        """
        computes UMass coherence scores for all latent topics

        input:
        - M (int): # of top words to consider when computing UMass score

        returns:
        - list of coherence scores for each latent topic
        """

        scores = []
        for top_words_in_topic in self.top_words_in_topics:
            scores.append(self._get_umass_coherence_metric(top_words_in_topic,
                                                           M))

        return scores


    def _tweet_tokenizer(self, tweet_text, stem=True):
        """
        helper function for _fit_tf_idf() that cleans (removes punctuation and
        stopwords) and tokenizes tweets; also creates reverse lookup table
        (self.stem_lookup_table) so stemmed words can later be mapped to full
        words

        input:
        - tweet_text (string): text of one tweet in corpus
        - stem (boolean): whether or not to stem

        returns:
        - lowercased, stemmed, cleaned, and tokenized version of tweet (list)
           NOTE:
              - all punctuation is removed except '#' to keep hashtags intact
              - words with length < 2 are removed
              - URLs are removed
        """

        tknzr = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)

        punctuation = string.punctuation.replace('#','')
        tokenized = tknzr.tokenize(tweet_text)
        tokenized = list(map(lambda x: x.strip(punctuation).replace("'",''),
                             tokenized)) #remove apostrophes in contractions
        tokenized = list(filter(lambda x: x not in punctuation and len(x) > 2
                                and 'http' not in x, tokenized))

        if stem:
            stemmer = PorterStemmer()
            tokenized_stemmed = []
            for word in tokenized:
                stem = stemmer.stem(word)
                tokenized_stemmed.append(stem)
                if stem not in self.stem_lookup_table:
                    self.stem_lookup_table[stem] = set([word])
                else:
                    self.stem_lookup_table[stem].add(word)

        return tokenized_stemmed


    def _get_stopwords(self):
        """
        helper function to get list of stopwords (standard english stopwords
        included in nltk.corpus library plus a few added stopwords)

        returns:
        - list of all stopwords
        """

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
        """
        helper function to get the most commonly occuring words in each topic

        input:
        - topN (int): # of most commonly occuring words to return

        returns:
        - list of the topN words in a topic
        """

        indices = np.array(np.flip(np.argsort(self.H, axis=1), axis=1))
        top_words_in_topics = []
        for row in indices:
            top_n_words = self.words[row][:topN]
            top_words_in_topics.append(top_n_words)

        return top_words_in_topics


    def _get_umass_coherence_metric(self, top_words_in_topic, M=5):
        """
        helper function for get_umass_coherence_scores() that simply computes
        the UMass coherence score for a single topic

        input:
        - top_words_in_topic (list of strings): top words in a particular topic
        - M (int): # of top words to consider when computing UMass score

        returns:
        - UMass coherence score (float) for a particular topic
        """

        terms_to_sum = []
        tf_idf = self.tf_idf
        tf_idf[tf_idf != 0] = 1

        for m in range(1, M):
            word_m = top_words_in_topic[m]
            word_m_index = self.vocab[word_m] # index of col corresponding to
                                              # this word in TF-IDF
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

    def get_tweets_in_each_topic(self):
        """ build a dict mapping topic indices (keys) to the corresponding
            tweets in that topic (values)

        returns:
        -  dict mapping of topic indices to the tweets most strongly associated
           with that topic
        """

        # each tweet can be associated with multiple topics, but here we
        # categorize each tweet based on the topic with which it is most
        # strongly associated
        indices = np.argmax(self.W, axis=1)
        tweets_in_each_topic = {}
        for tweet_idx, topic_idx in enumerate(indices):
            if topic_idx not in tweets_in_each_topic:
                tweets_in_each_topic[topic_idx] = [self.tweet_corpus[tweet_idx]]
            else:
                tweets_in_each_topic[topic_idx] += [self.tweet_corpus[tweet_idx]]

        return tweets_in_each_topic


if __name__ == '__main__':
    
    tc = TweetCorpus(['../data/03_28_2018_18_02.pkl',
                     '../data/03_30_2018_15_37.pkl'])


    nmf_mod = BuildNMF(tc.hashtag_aggregated_corpus, num_topics=50)
    nmf = nmf_mod.fit()
