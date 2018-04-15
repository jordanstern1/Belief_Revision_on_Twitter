from get_tweet_corpus import TweetCorpus
from BuildNMF import BuildNMF
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pickle
from get_tweets import unpickle_results
import pdb


def plot_reconstruction_err(tweet_corpus, num_topics_list, max_iter=100,
                            savepath=None, show=False):
    """ create scatter plot of reconstruction error from NMF vs. num. topics
        (reconstruction err = Frobenius norm of (TF-IDF - WH))

    input:
    - tweet_corpus (list or np array of strings):
        corpus of text content from tweets
    - num_topics_list (list of ints):
        list of numbers of latent topics in NMF
        (reconstruction error is computed for each value in num_topics_list)
    - max_iter (int):
        max number of iterations in NMF computation
    - savepath (string):
        file path where plot should be saved
    - show (boolean):
        if True, then plot is displayed

    output:
    - displays plot if show == True

    returns:
    - list of reconstruction_errs  for each num in num_topics_list

    """

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
                         show=False, save=True, M=5):
    """ creates histograms of UMass coherence scores for each num in
        num_topics list

    input:
    - tweet_corpus (list or np array of strings):
        corpus of text content from tweets
    - num_topics_list (list of ints):
        list of numbers of latent topics in NMF
    - max_iter (int):
        max number of iterations in NMF computation
    - save (string):
        if True, then histograms are saved
    - show (boolean):
        if True, then histograms are displayed
    - M (int):
        number of most commonly occuring words in a topic to consider when
        computing UMass coherence scores for each topic

    output:
    - displays plot if show == True

    returns:
    - None

    """

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

        plt.savefig('../plots/coherence_scores_' + str(num) + '_topics.png')

        if show:
            plt.show()


def plot_mean_umass_scores(tweet_corpus, num_topics_list, max_iter=100,
                           savepath=None, show=False, M=5):
    """ creates a bar chart of mean UMass coherence scores for each num in
        num_topics list

    input:
    - tweet_corpus (list or np array of strings):
        corpus of text content from tweets
    - num_topics_list (list of ints):
        list of numbers of latent topics in NMF
    - max_iter (int):
        max number of iterations in NMF computation
    - savepath (string):
        file path where plot should be saved
    - show (boolean):
        if True, then histograms are displayed
    - M (int):
        number of most commonly occuring words in a topic to consider when
        computing UMass coherence scores for each topic

    output:
    - displays plot if show == True

    returns:
    - None

    """

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

    if savepath:
        plt.savefig(savepath)

    if show:
        plt.show()


def umass_box_and_whiskers_for_diff_copora(tc, num_topics, max_iter=100,
                                           savepath=None, show=True, M=5):
    """ creates a plot with a box and whiskers to display the distribution
        of UMass coherence scores for each type of tweet corpus

    input:
    - tc (TweetCorpus object):
        contains different corpora as attributes
    - num_topics_list (list of ints):
        list of numbers of latent topics in NMF
    - max_iter (int):
        max number of iterations in NMF computation
    - savepath (string):
        file path where plot should be saved
    - show (boolean):
        if True, then histograms are displayed
    - M (int):
        number of most commonly occuring words in a topic to consider when
        computing UMass coherence scores for each topic

    output:
    - displays plot if show == True

    returns:
    - None

    """

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
    plt.title('Coherence Score Boxplots\n(Number of Topics = {})'\
                                    .format(num_topics), size=16)
    plt.xlabel('Tweet Corpus Used', size=14)
    plt.ylabel('UMass Coherence Score', size=14)
    sns.boxplot(x=corpora_labels, y=scores)

    if show:
        plt.show()

    if savepath:
        plt.savefig(savepath)


def umass_box_and_whiskers_for_diff_num_topics(tweet_corpus, num_topics_list,
                                               max_iter=100, savepath=None,
                                               show=True, M=5):
    """ creates a plot with a box and whiskers to display the distribution
        of UMass coherence scores for different numbers of latent topics in NMF

    input:
    - tweet_corpus (list or np array of strings):
        corpus of text content from tweets
    - num_topics_list (list of ints):
        list of numbers of latent topics in NMF
    - max_iter (int):
        max number of iterations in NMF computation
    - savepath (string):
        file path where plot should be saved
    - show (boolean):
        if True, then histograms are displayed
    - M (int):
        number of most commonly occuring words in a topic to consider when
        computing UMass coherence scores for each topic

    output:
    - displays plot if show == True

    returns:
    - None

    """

    scores_list = []
    for num in num_topics_list:
        nmf_mod = BuildNMF(tweet_corpus, num_topics=num)
        nmf_mod.fit(max_iter=max_iter, display=False)
        coherence_scores = nmf_mod.get_umass_coherence_scores(M)
        scores_list.append(coherence_scores)


    plt.subplots(5,2,figsize=[14,14])
    i = 1
    for num_topics, scores in zip(num_topics_list,scores_list):
        plt.subplot(2,5,i)
        plt.ylabel('UMass Coherence Score', size=10)
        plt.title('Number of Topics = {}'.format(num_topics), size=12)
        sns.boxplot(y=scores)
        i+=1
    plt.tight_layout()

    if show:
        plt.show()

    if savepath:
        plt.savefig(savepath)


def get_bar_chart_of_topic_sizes(nmf_mod, title, ylabel, show=True,
                                savepath=None):
    """ plot bar chart of topic sizes

    input:
    - nmf_mod (BuildNMF object):
        fit NMF model containing a get_tweets_in_each_topic() method
    - title (string): title of plot
    - ylabel (string): y-axis label
    - show (boolean): if True, then histograms are displayed
    - savepath (string): file path where plot should be saved

    output:
    - displays plot if show == True

    returns:
    - None
    """

    tweets_by_topic = nmf_mod.get_tweets_in_each_topic()
    num_topics = nmf_mod.num_topics
    topic_sizes = []
    for i in range(num_topics):
        topic_sizes.append(len(tweets_by_topic[i]))
    plt.figure(figsize=(10,10))
    plt.bar(np.arange(num_topics), topic_sizes)
    plt.xlabel('Topic #', size=14)
    plt.ylabel(ylabel, size=14)
    plt.title(title, size=16)

    if show:
        plt.show()
    if savepath:
        plt.savefig(savepath)


if __name__ == '__main__':

    pickled_tweet_batches = ['../data/03_28_2018_18_02.pkl',
                             '../data/03_30_2018_15_37.pkl']
    tc = TweetCorpus(pickled_tweet_batches)

    plt.close('all')


    # Plot bar chart of topic sizes
    # ideal_num_topics = 50 # determined from coherence score boxplots
    # nmf_mod = BuildNMF(tc.hashtag_aggregated_corpus, num_topics=ideal_num_topics)
    # nmf = nmf_mod.fit()
    # get_bar_chart_of_topic_sizes(nmf_mod, 'Sizes of Tweet Topics',
    #                             'Number of Tweets in Topic',
    #                             savepath='../plots/tweet_topic_sizes.png')


    # Box and whiskers of topic coherences for different numbers of topics
    num_topics_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    corp = tc.hashtag_aggregated_corpus
    score = umass_box_and_whiskers_for_diff_num_topics(corp, num_topics_list,
         savepath='../plots/coherence_score_boxplots_for_diff_num_topics2.png')



    # Box and whiskers of topic coherences for different corpora
    # (raw corpus, quote-aggregated, and hashtag-aggregated)
    # umass_box_and_whiskers_for_diff_copora(tc, 15,
    # savepath='../plots/coherence_score_boxplots_diff_corpora.png', max_iter=200, M=5)

    # num_topics_list = [5, 10, 12, 15, 20, 30, 40, 50, 60, 80, 100]
    # plot_mean_umass_scores(tc.hashtag_aggregated_corpus, num_topics_list,
    #                         savepath='../plots/coherence_score_means.png')


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
    #                                 savepath='../plots/reconstruction_err3.png')
