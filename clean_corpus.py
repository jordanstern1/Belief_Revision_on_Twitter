from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.stem.porter import *
import sklearn.decomposition
import string
from nltk import word_tokenize
from nltk.corpus import stopwords
import pdb

from nltk.tokenize import TweetTokenizer

from get_tweets import unpickle_results

def load_tweet_corpus(fname):
    tweets = unpickle_results(fname)
    tweet_text = [t.all_text for t in tweets]
    tweet_corpus = list(set(tweet_text)) # remove duplicates

    return tweet_corpus


def clean_tokenize_stem(corpus, stem=False):
    '''
    input: np array where each row is a string containing all text from each doc
    in the corpusd
    output: same as input, but all characters have been lowercased, stopwords
    have been removed, and words have been stemmed
    '''
    cleaned_corpus = []
    for doc in corpus:
        # lowercase and remove punc
        lowered_str_no_punc = " ".join([word.strip(punctuation)
                              for word in doc.lower().split()])

        # tokenize
        tokenized = word_tokenize(lowered_str_no_punc)

        # remove stopwords
        additional_stopwords = []
        stopwordz = set(stopwords.words('english') + additional_stopwords)
        tokenized_no_stopwords = [word for word in tokenized if word not in stopwordz]

        if stem:
            stemmer = PorterStemmer()
            result = " ".join([stemmer.stem(word) for word in tokenized_no_stopwords])
            cleaned_corpus.append(result)
        else:
            cleaned_corpus.append(tokenized_no_stopwords)

    return cleaned_corpus


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

def tweet_tokenizer(tweet_text):
    tknzr = TweetTokenizer(preserve_case=False, strip_handles=True,
                           reduce_len=True)
    #from string import punctuation
    punctuation = string.cpunctuation.replace('#','')
    tokenized = tknzr.tokenize(tweet_text)
    tokenized = list(map(lambda x: x.strip(punctuation).replace("'",''),
                         tokenized))
    tokenized = list(filter(lambda x: x not in punctuation and len(x) > 1
                            and 'http' not in x, tokenized))

    return tokenized

if __name__ == '__main__':
    tweet_corpus = load_tweet_corpus('data/query_results_03_26_2018_1.pkl')
    cleaned_tokenized_corpus = clean_tokenize(tweet_corpus)
    #cleaned_tweet = tweet_tokenizer(tweet_corpus[101])
