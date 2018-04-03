import numpy as np
import sklearn.decomposition
from clean_corpus import *
import matplotlib.pyplot as plt
import seaborn as sns
from get_tweets import pickle_results
import time

def error(V, W, H):
    '''
    Goal: This is a method that uses W, H, and the document matrix (V) to calculate MSE
    input: none
    output: return the reconstructon error after NMF
    '''
    return np.linalg.norm(V - W.dot(H))


def get_NMF(V, words=None, num_topics=25, max_iter=100, solver='cd', topN=20, display=True):
    '''
    input: V matrix to be factorized by NMF, number of topics for NMF, max
    number of iterations for NMF factorization
    output: prints topN words in each topic if display=True
    returns: reconstruction error from NMF factorization
    '''
    print('solver = ', solver)
    nmf = sklearn.decomposition.NMF(n_components=num_topics,
                                       max_iter=max_iter,
                                       solver=solver)
    W = nmf.fit_transform(V)
    err = nmf.reconstruction_err_
    H = nmf.components_

    if display:
        top_words_in_topics = get_top_words_in_topics(H, words=words, topN=topN)
        for idx, topic in enumerate(top_words_in_topics):
            print('-'*20)
            print('Most common words for Topic', idx, ':')
            for word in topic:
                print(word)

    return nmf

def plot_reconstruction_err(V, num_topics, savepath='plots/reconstruction_err.eps',
                            pickle_path='reconstruction_errs.pkl'):

    reconstruction_errs = []

    for num in num_topics:
        nmf = get_NMF(V, num_topics=num, solver='mu', display=False)
        reconstruction_errs.append(nmf.reconstruction_err_)

    plt.figure()
    plt.title('NMF Reconstruction Error vs. Number of Topics \n (Max Number of Iterations = 200)')
    plt.xlabel('Number of Topics')
    plt.ylabel('Reconstruction Error')
    plt.scatter(num_topics, reconstruction_errs)
    plt.savefig(savepath)
    plt.show()

    return reconstruction_errs


def get_top_words_in_topics(H, words, topN=10):
    indices = np.array(np.flip(np.argsort(H, axis=1), axis=1))
    top_words_in_topics = []
    for row in indices:
        top_n_words = words[row][:topN]
        top_words_in_topics.append(top_n_words)

    return top_words_in_topics



def get_umass_coherence_metric(corpus_tf_idf, vec, top_words_in_topic, M=10):
    vocab = vec.vocabulary_
    terms_to_sum = []
    corpus_tf_idf[corpus_tf_idf != 0] = 1

    for m in range(1, M):
        word_m = top_words_in_topic[m]
        word_m_index = vocab[word_m] # index of col corresp. to this word in tf-idf
        tfidf_column_for_word_m = corpus_tf_idf[:,word_m_index].todense()\
                                                               .astype(int)
        for l in range(0, m-1):
            word_l = top_words_in_topic[l]
            word_l_index = vocab[word_l]
            num_docs_with_word_l = corpus_tf_idf[:,word_l_index].sum()
            tfidf_column_for_word_l = corpus_tf_idf[:,word_l_index].todense()\
                                                                   .astype(int)
            num_docs_with_both_words = (tfidf_column_for_word_m
                                        & tfidf_column_for_word_l).sum()
            result = (num_docs_with_both_words + 1)/num_docs_with_word_l
            terms_to_sum.append(result)

    coherence_score = np.sum(np.log(terms_to_sum))
    return coherence_score


def plot_mean_coherence_scores():
    # scores = [get_umass_coherence_metric(corpus_tf_idf, vec,/
    #           top_words_in_topic, M=10) for ]
    pass



if __name__ == '__main__':
    raw_tweet_corpus = load_tweet_corpus2(['../data/03_28_2018_18_02.pkl',
                                           '../data/03_30_2018_15_37.pkl'])
    vec, corpus_tf_idf = get_tf_idf(raw_tweet_corpus)
    words = np.array(vec.get_feature_names())
    print('Text pre-processing complete, now computing NMF')
    tic = time.time()
    nmf = get_NMF(corpus_tf_idf.todense(), words, num_topics=15, solver='mu',
                  max_iter=200)
    toc = time.time()
    print('time elapsed = {:.3f}'.format(toc-tic))


    # Plot reconstruction error
    # num_topics = [3,5,7,10,12,15,18,20,50, 100, 200]
    # errors = plot_reconstruction_err(corpus_tf_idf.todense(),
    #                                  num_topics,
    #                                  savepath='../plots/reconstruction_err2.eps')
