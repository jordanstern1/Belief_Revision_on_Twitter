from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from nltk.stem.porter import *
import sklearn.decomposition
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
import pdb
import numpy as np
from nltk.tokenize import TweetTokenizer
from get_tweets import unpickle_results
from collections import Counter, OrderedDict


def load_tweet_corpus(fname):
    tweets = unpickle_results(fname)
    tweet_text = [t.all_text for t in tweets]
    tweet_corpus = np.array(list((set(tweet_text)))) # remove duplicates

    return tweet_corpus


def clean_tokenize(tweet_corpus):
    '''
    input:
    output:
    returns:
    '''

    cleaned_tokenized_corpus = []

    for tweet in tweet_corpus:
        tokenized = tweet_tokenizer(tweet)
        cleaned_tokenized_corpus.append(tokenized)

    return cleaned_tokenized_corpus


def tweet_tokenizer(tweet_text, stem=True):
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True,
                           reduce_len=True)

    punctuation = string.punctuation.replace('#','')
    tokenized = tknzr.tokenize(tweet_text)
    tokenized = list(map(lambda x: x.strip(punctuation).replace("'",''),
                         tokenized))
    tokenized = list(filter(lambda x: x not in punctuation and len(x) > 1
                            and 'http' not in x, tokenized))

    if stem:
        stemmer = PorterStemmer()
        tokenized = [stemmer.stem(word) for word in tokenized]

    return tokenized

def get_stopwords():
    # get standard English stopwords and remove internal apostrophes
    english_stopwords = stopwords.words('english')
    english_stopwords = list(map(lambda x: x.replace("'",''), english_stopwords))

    # add stopwords
    additional_stopwords = ['changed', 'my', 'mind', 'view','opinion']
    stemmer = PorterStemmer()
    additional_stopwords = [stemmer.stem(word) for word in additional_stopwords]
    all_stopwords =  english_stopwords + additional_stopwords

    return all_stopwords

def get_tf_idf(raw_tweet_corpus):

    stopwords = get_stopwords()

    vec = TfidfVectorizer(strip_accents='ascii',tokenizer=tweet_tokenizer,
                          stop_words=stopwords, max_df=0.5, max_features=None)
    corpus_tf_idf = vec.fit_transform(raw_tweet_corpus)

    return vec, corpus_tf_idf


def get_tf_matrix(raw_tweet_corpus):
    """ get bag of words """

    stopwords = get_stopwords()

    cv = CountVectorizer(strip_accents='ascii',tokenizer=tweet_tokenizer,
                          stop_words=stopwords, max_df=0.5, max_features=None)
    corpus_tf_mat = cv.fit_transform(raw_tweet_corpus)

    return cv, corpus_tf_mat


def get_most_common_words(raw_tweet_corpus):
    cv, corpus_tf_mat = get_tf_matrix(raw_tweet_corpus)
    words = cv.get_feature_names()
    term_frequencies = corpus_tf_mat.sum(axis=0).tolist()[0]

    result = Counter(dict(zip(words,term_frequencies))).most_common()

    return result

if __name__ == '__main__':
    raw_tweet_corpus = load_tweet_corpus('data/03_28_2018_18_02.pkl')

    most_common = get_most_common_words(raw_tweet_corpus)

    # vocab = Counter(vec.vocabulary_)
    # most_common_words = vocab.most_common()
