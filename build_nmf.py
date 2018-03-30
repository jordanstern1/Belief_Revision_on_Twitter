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
    nmf_sk = sklearn.decomposition.NMF(n_components=num_topics,
                                       max_iter=max_iter,
                                       solver=solver)
    W_sk = nmf_sk.fit_transform(V)
    err = nmf_sk.reconstruction_err_
    H_sk = nmf_sk.components_

    indices_sk = np.array(np.flip(np.argsort(H_sk, axis=1), axis=1))
    #print most common words by topic
    if display:
        for idx, row in enumerate(indices_sk):
            #pdb.set_trace()
            print('-'*20)
            print('Most common words for Topic', idx, ':')
            top_n_words = words[row][:topN]
            for word in top_n_words:
                print(word)

    return err

def plot_reconstruction_err(V, num_topics, savepath='plots/reconstruction_err.eps',
                            pickle_path='reconstruction_errs.pkl'):

    reconstruction_errs = [get_NMF(V, num_topics=num, solver='mu', display=False)
                           for num in num_topics]

    plt.figure()
    plt.title('NMF Reconstruction Error vs. Number of Topics \n (Max Number of Iterations = 100)')
    plt.xlabel('Number of Topics')
    plt.ylabel('Reconstruction Error')
    plt.scatter(num_topics, reconstruction_errs)
    plt.savefig(savepath)
    plt.show()

    return reconstruction_errs


if __name__ == '__main__':
    raw_tweet_corpus = load_tweet_corpus2(['data/03_28_2018_18_02.pkl',
                                           'data/03_30_2018_15_37.pkl'])
    vec, corpus_tf_idf = get_tf_idf(raw_tweet_corpus)
    words = np.array(vec.get_feature_names())
    print('Text pre-processing complete, now computing NMF')
    # tic = time.time()
    # err = get_NMF(corpus_tf_idf.todense(), words, num_topics=15, solver='cd',
    #               max_iter=200)
    # toc = time.time()
    # print('time elapsed = {:.3f}'.format(toc-tic))


    num_topics = [3,5,7,10,12,15,18,20,50]
    errors = plot_reconstruction_err(corpus_tf_idf.todense(),
                                     num_topics,
                                     savepath='plots/reconstruction_err2.eps')
