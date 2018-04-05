from get_tweet_corpus import TweetCorpus
from BuildNMF import BuildNMF
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_reconstruction_err(tweet_corpus, num_topics_list, max_iter=100,
                            savepath=None, show=False):

    reconstruction_errs = []

    for num in num_topics_list:
        nmf_mod = BuildNMF(tweet_corpus, num_topics=num)
        nmf_mod.fit(max_iter=max_iter, display=False)
        reconstruction_errs.append(nmf_mod.reconstruction_err)

    plt.figure()
    plt.title('NMF Reconstruction Error vs. Number of Topics\n' +
              '(Max Number of Iterations = {})'.format(max_iter))
    plt.xlabel('Number of Topics')
    plt.ylabel('Reconstruction Error')
    plt.scatter(num_topics_list, reconstruction_errs)

    if savepath:
        plt.savefig(savepath)

    if show:
        plt.show()

    return reconstruction_errs


def get_umass_histograms(tweet_corpus, num_topics_list, max_iter=100,
                         show=False, M=10):

    for num in num_topics_list:
        nmf_mod = BuildNMF(tweet_corpus, num_topics=num)
        nmf_mod.fit(max_iter=max_iter, display=False)
        coherence_scores = nmf_mod.get_umass_coherence_scores(M)

        plt.figure()
        plt.title('UMass Coherence Scores for Each Topic\n' +
                  '(Total Number of Topics = {})'.format(num))
        plt.xlabel('Topic #')
        plt.ylabel('UMass Coherence Score')
        plt.bar(np.arange(1, num + 1), coherence_scores)

        plt.savefig('../plots/coherence_scores2_' + str(num) + '_topics.png')

        if show:
            plt.show()


def plot_mean_umass_scores(tweet_corpus, num_topics_list, max_iter=100,
                           show=False, M=10):

    mean_scores = []

    for num in num_topics_list:
        nmf_mod = BuildNMF(tweet_corpus, num_topics=num)
        nmf_mod.fit(max_iter=max_iter, display=False)
        coherence_scores = np.array(nmf_mod.get_umass_coherence_scores(M))
        mean_scores += [coherence_scores.mean()]

    plt.figure()
    plt.title('Mean UMass Coherence Scores for Different Numbers of Topics')
    plt.xlabel('Number of Topics')
    plt.ylabel('Mean UMass Coherence Score Across All Topics')
    plt.bar(num_topics_list, mean_scores)

    plt.savefig('../plots/coherence_score_means2.png')

    if show:
        plt.show()


def get_umass_box_and_whiskers(tc, num_topics, max_iter=100, M=10,
                               savepath=None, show=True):
    """ Description here """

    corpora = [tc.raw_tweet_corpus, tc.quote_aggregated_corpus,
               tc.hashtag_aggregated_corpus]
    corpora_labels = ['Raw Tweet Corpus', 'Quote-aggregated\nCorpus',
                      'Hashtag-aggregated\nCorpus']

    scores = []
    for corpus in corpora:
        nmf_mod = BuildNMF(corpus, num_topics=num_topics)
        nmf_mod.fit(max_iter=max_iter, display=False)
        scores.append(nmf_mod.get_umass_coherence_scores(M))

    plt.figure(figsize=(10,10))
    plt.title('Coherence Score Boxplots', size=16)
    plt.xlabel('Tweet Corpus Used', size=14)
    plt.ylabel('UMass Coherence Score', size=14)
    sns.boxplot(x=corpora_labels, y=scores)

    if show:
        plt.show()

    if savepath:
        plt.savefig(savepath)


if __name__ == '__main__':

    pickled_tweet_batches = ['../data/03_28_2018_18_02.pkl',
                             '../data/03_30_2018_15_37.pkl']
    tc = TweetCorpus(pickled_tweet_batches)

    plt.close('all')
    get_umass_box_and_whiskers(tc, 15,
                               savepath='../plots/coherence_score_boxplots.png',
                               max_iter=200, M=10)







    ####################################################################

    # Plot mean UMass coherence scores for different numbers of topics
    # tweet_corpus = tc.raw_tweet_corpus
    # num_topics_list = [5, 10, 12, 15, 20, 30, 40, 50, 60, 80, 100]
    # num_topics_list = np.arange(2,121)
    # plot_mean_umass_scores(tweet_corpus, num_topics_list, max_iter=100,
    #                            show=False)

    # Plot coherence scores
    # num_topics_list=[5]
    # num_topics_list = [5, 10, 12, 15, 20, 30, 40, 50, 60, 80, 100]
    # get_umass_histograms(tweet_corpus, num_topics_list)


    # Plot reconstruction error
    # num_topics_list = [3,5,7,10]
    # errors = plot_reconstruction_err(tweet_corpus, num_topics_list,
    #                                 savepath='../plots/reconstruction_err3.eps')
