from get_tweet_corpus import TweetCorpus
import pandas as pd
import matplotlib.pyplot as plt


def plot_time_series(tc, token, show=True, savepath=None):
    """ plot a time series of daily counts of tweets containing a certain token

    input:
    - tc (TweetCorpus object):
        contains pandas dataframe as an attribute (note about dataframe: 'text'
        column contains the original text of each tweet in the corpus, 'date'
        column contains the date when the tweet was authored)
    - token (string):
        function will plot time series of daily counts of tweets containing this
        particular token
    - show (boolean): if True, plot is displayed
    - savepath (string): file path where plot should be saved

    output:
    - plot is displayed (if savepath == True)
    - plot is saved (if savepath is provided)

    """

    # plot a time series of counts by day
    # NOTE: 'D' = day
    new_df = tc.tweet_df.copy()
    new_df = new_df[new_df.apply(lambda x: token in x['text'].lower(), axis=1)]
    new_df[['text']].resample('D')\
                    .count()\
                    .plot(figsize=(10,10), legend=False)
    plt.xlabel('Day', size=14)
    plt.ylabel('# of Tweets Per Day', size=14)
    plt.title('Daily Counts of Tweets Containing "{}"'.format(token), size=16)

    if show:
        plt.show()
    if  savepath:
        plt.savefig(savepath)


if __name__ == '__main__':
    tc = TweetCorpus(['../data/03_28_2018_18_02.pkl',
                      '../data/03_30_2018_15_37.pkl'])

    plt.close('all')
    # Plot time series of daily counts of tweets containing these
    # words (words associated w/ most coherent topics)
    tokens = ['daca', 'xbox', 'brexit', 'valentine', 'gun', 'love', 'netflix',
              'baby', 'nuclear', 'marry']
    for token in tokens:
        plot_time_series(tc, token, show=False,
                         savepath='../plots/time_series_{}.png'.format(token))

    # ideal_num_topics = 50 # determined from coherence score boxplots
    # nmf_mod = BuildNMF(tc.hashtag_aggregated_corpus,
    #                    num_topics=ideal_num_topics)
    #
    # tweets_by_topic = nmf_mod.get_tweets_in_each_topic()
