from wordcloud import WordCloud
from get_tweet_corpus import TweetCorpus
from BuildNMF import BuildNMF
import matplotlib.pyplot as plt
import numpy as np
import os


def topic_word_frequency(nmf_mod, topic_idx):
    ''' get dict w/ words in corpus as keys and their frequencies as values

    input:
    - nmf_mod (BuildNMF object): this should be a fitted BuildNMF object
    - topic_idx (int): index of the topic for which we want to build the dict

    returns:
    - dict w/ words in corpus as keys and their frequencies as values
      (since words were stemmed during TF-IDF construction, we also map
      each stemmed word back to a full word)

    '''

    tot = np.sum(nmf_mod.H[topic_idx])
    frequencies = [val / tot for val in nmf_mod.H[topic_idx]]

    # convert stems back to full words
    #(pick the shortest word of all the words
    # in the corpus that share the same stem)
    unstemmed_words = []
    for word in nmf_mod.words:
        unstemmed = min(list(nmf_mod.stem_lookup_table[word]), key=len)
        unstemmed_words.append(unstemmed)

    return dict(zip(unstemmed_words, frequencies))


def topic_word_cloud(nmf_mod, topic_idx, max_words=200, figsize=(14, 8),
                     width=2400, height=1300, ax=None, show=False,
                     savepath=None):
    ''' create wordcloud for a particular topic

    input:
    - nmf_mod (BuildNMF object): this should be a fitted BuildNMF object
    - topic_idx (int): index of the topic used to generate wordcloud
    - max_words (int): maximum number of words in wordcloud
    - figsize (tuple): size of the matplotlib figure that displays the wordcloud
    - width (int): width of the wordcloud image
    - height (int): height of the wordcloud image
    - ax (matplotlib axes): axes to use to display wordcloud
    - show (boolean): if True then wordcloud is displayed
    - savepath (string): file path where wordcloud image will be saved

    output:
    - if show == True, displays the wordcloud

    returns:
    - None

    '''

    wc = WordCloud(background_color='white', max_words=max_words, width=width,
                   height=height)
    word_freq = topic_word_frequency(nmf_mod, topic_idx)

    # Fit WordCloud object
    wc.fit_words(word_freq)

    if not ax:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
    ax.imshow(wc)
    ax.axis('off')

    if show:
        plt.show()

    if savepath:
        plt.savefig(savepath)

    plt.close('all')


def gen_wordclouds_for_all_topics(nmf_mod):
    """ generate wordclouds for all topics

    input:
    - nmf_mod (BuildNMF object): this should be a fitted BuildNMF object

    output:
    - wordcloud images are saved to directory images_##topics/ with the names
      'wordcloud_ranking#_topic#' where the ranking # corresponds to the UMass
      coherence score of that topic (ranking of zero is the most coherent),
      and topic # is the index of that topic

    returns:
    - None

    """

    # generate wordclouds for most coherent topics
    # NOTE: UMass scores are always negative and the closer a UMass score gets
    # to zero, the more coherent that topic is
    coherence_scores = np.abs(np.array(nmf_mod.get_umass_coherence_scores()))
    indices = np.argsort(coherence_scores)
    coherence_scores_sorted = coherence_scores[indices]

    directory = '../pres_wordclouds_' + str(nmf_mod.num_topics)+ 'topics/'
    if not os.path.exists(directory):
        os.makedirs(directory)

    for ranking, idx in enumerate(indices):
        topic_word_cloud(nmf_mod, idx,
                        savepath= directory + 'wordcloud_ranking'+ str(ranking)\
                        + '_topic' + str(idx) + '.png')


if __name__ == '__main__':
    tc = TweetCorpus(['../data/03_28_2018_18_02.pkl',
                      '../data/03_30_2018_15_37.pkl'])


    # Get wordclouds for all topics in tweet corpus
    ideal_num_topics = 50 # determined from coherence score boxplots
    nmf_mod = BuildNMF(tc.hashtag_aggregated_corpus,
                       num_topics=ideal_num_topics)
    nmf = nmf_mod.fit(display=True)

    gen_wordclouds_for_all_topics(nmf_mod)
