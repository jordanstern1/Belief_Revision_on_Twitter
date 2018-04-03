from BuildNMF import BuildNMF, load_tweet_corpus
import matplotlib.pyplot as plt
import numpy as np

def plot_reconstruction_err(raw_tweet_corpus, num_topics_list, max_iter=100,
                            savepath=None, show=False):

    reconstruction_errs = []

    for num in num_topics_list:
        nmf_mod = BuildNMF(raw_tweet_corpus, num_topics=num)
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


def get_umass_histograms(raw_tweet_corpus, num_topics_list, max_iter=100,
                         show=False):

    for num in num_topics_list:
        nmf_mod = BuildNMF(raw_tweet_corpus, num_topics=num)
        nmf_mod.fit(max_iter=max_iter, display=False)
        coherence_scores = nmf_mod.get_umass_coherence_scores()

        plt.figure()
        plt.title('UMass Coherence Scores for Each Topic\n' +
                  '(Total Number of Topics = {})'.format(num))
        plt.xlabel('Topic #')
        plt.ylabel('UMass Coherence Score')
        plt.bar(np.arange(1, num + 1), coherence_scores)

        plt.savefig('../plots/coherence_scores_' + str(num) + '_topics.eps')

        if show:
            plt.show()




if __name__ == '__main__':
    raw_tweet_corpus = load_tweet_corpus(['../data/03_28_2018_18_02.pkl',
                                          '../data/03_30_2018_15_37.pkl'])

    plt.close('all')

    # Plot coherence scores
    num_topics_list = [5, 10, 12, 15, 20, 30, 50, 100]
    get_umass_histograms(raw_tweet_corpus, num_topics_list)


    # Plot reconstruction error
    # num_topics_list = [3,5,7,10]
    # errors = plot_reconstruction_err(raw_tweet_corpus, num_topics_list,
    #                                 savepath='../plots/reconstruction_err3.eps')
