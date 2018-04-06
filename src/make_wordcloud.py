
from wordcloud import WordCloud
from get_tweet_corpus import TweetCorpus
from BuildNMF import BuildNMF
import matplotlib.pyplot as plt
import numpy as np


def topic_word_frequency(nmf_mod, topic_idx):
    ''' Return (word, frequency) tuples for creating word cloud
    INPUT:
        topic_idx: int
    '''
    tot = np.sum(nmf_mod.H[topic_idx])
    frequencies = [val / tot for val in nmf_mod.H[topic_idx]]

    # convert stems back to full words (pick the shortest word of all the words
    # inthe corpus that share the same stem)
    unstemmed_words = []
    for word in nmf_mod.words:
        unstemmed = min(list(nmf_mod.stem_lookup_table[word]), key=len)
        unstemmed_words.append(unstemmed)

    return dict(zip(unstemmed_words, frequencies))


def topic_word_cloud(nmf_mod, topic_idx, max_words=200, figsize=(14, 8),
                     width=2400, height=1300, ax=None, show=False,
                     savepath=None):
    ''' create word cloud for a given topic

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


def gen_wordclouds_for_most_coherent_topics(nmf_mod, num_topics):

    # generate wordclouds for most coherent topics
    coherence_scores = np.abs(np.array(nmf_mod.get_umass_coherence_scores()))
    tenth_percentile = np.percentile(coherence_scores,10)
    indices = np.arange(ideal_num_topics)[coherence_scores <= tenth_percentile]
    for idx in indices:
        topic_word_cloud(nmf_mod, idx,
                         savepath='../images/wordcloud_topic'+str(idx)+'.eps')


if __name__ == '__main__':
    tc = TweetCorpus(['../data/03_28_2018_18_02.pkl',
                      '../data/03_30_2018_15_37.pkl'])


    ideal_num_topics = 50 # determined from coherence score boxplots
    nmf_mod = BuildNMF(tc.hashtag_aggregated_corpus,
                       num_topics=ideal_num_topics)
    nmf = nmf_mod.fit(display=True)

    gen_wordclouds_for_most_coherent_topics(nmf_mod, ideal_num_topics)




    #topic_word_cloud(nmf_mod, 0)
