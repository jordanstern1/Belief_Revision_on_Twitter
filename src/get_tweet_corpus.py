from get_tweets import unpickle_results
import pandas as pd
import numpy as np


class TweetCorpus(object):
    """
    Given a list of .pkl files containing lists of tweet objects, this class
    will recover the tweet objects from the .pkl files and assemble various
    corpora of tweets (stored as class attributes) that can then be used for
    topic modeling using the BuildNMF class. The class also constructs a pandas
    dataframe from the tweet objects, which is useful for exploratory data
    analysis.

    Methods
    --------
    Note: all methods are helper functions used to build the tweet corpora and
    the pandas dataframe

    - _get_corpus():
        recovers tweet objects from .pkl files, & builds three types of corpora
        (raw_tweet_corpus, quote_aggregated_corpus, and hashtag_aggregated_corpus)
    - _aggregate_by_hashtag():
        aggreggates tweets with the same hashtag to synthesize longer documents
        for improved topic modeling
    - _create_tweet_df():
        constructs a pandas dataframe that contains useful information about
        each tweet in the corpus

    Attributes
    -----------
    - pickled_tweet_batches:
        list of .pkl file names storing lists of tweet objects
    - tweet_object_list:
        list of tweet objects recovered from .pkl files
    - raw_tweet_corpus:
        np array of text content from each tweet (no aggregation)
    - quote_aggregated_corpus:
        np array of text content from each tweet where quote tweets; when we
        encounter a quote tweet (tweet that is commenting on a quoted tweet),
        we combine the text of the comment and the quoted tweet for added
        context
    - hashtag_aggregated_corpus:
        np array of tweets that have been aggregated by hashtag (note: this
        corpus also aggregates quote tweets)
    - tweet_df:
        pandas dataframe containing the following information about each tweet
        in the corpus: date of tweet, text of tweet, Twitter user info
        (username, user bio, profile location), text of quoted tweet,
        at-mentions, hashtags, urls included in tweet, tweet type('tweet' or
        'quote'), geo coordinates of location where tweet was authored, &
        unique id of that particular tweet

    """

    def __init__(self, pickled_tweet_batches):
        self.pickled_tweet_batches = pickled_tweet_batches
        self.tweet_object_list = self._get_corpus('objects')

        self.raw_tweet_corpus = self._get_corpus('raw_text')
        self.quote_aggregated_corpus = self._get_corpus('aggregate_quote_tweets')
        self.hashtag_aggregated_corpus = self._get_corpus('aggregate_by_hashtag')
        self.tweet_df = self._create_tweet_df()


    def _get_corpus(self, option):
        """ recovers tweet objects from .pkl files, & builds three types of
            corpora (raw_tweet_corpus, quote_aggregated_corpus, and
            hashtag_aggregated_corpus)
        """

        all_tweets = []
        for fname in self.pickled_tweet_batches:
            all_tweets += unpickle_results(fname)

        if option == 'objects':
            return all_tweets

        if option == 'raw_text':
            tweet_text = [t.all_text for t in all_tweets]
            tweet_corpus = np.array(list((set(tweet_text)))) # remove duplicates
            return tweet_corpus

        if option == 'aggregate_quote_tweets':
            tweet_text = [t.quote_or_rt_text + ' ' + t.all_text for t in all_tweets]
            tweet_corpus = np.array(list((set(tweet_text)))) # remove duplicates
            return tweet_corpus

        if option == 'aggregate_by_hashtag':
            tweet_corpus = self._aggregate_by_hashtag()
            return tweet_corpus


    def _aggregate_by_hashtag(self):
        """ aggreggates tweets with the same hashtag to synthesize longer
            documents for improved topic modeling
        """

        hashtag_dict = {}
        tweets_without_hashtags = []

        for tweet in self.tweet_object_list:
            hashtags = tweet.hashtags
            if len(hashtags) == 0:
                tweets_without_hashtags += [tweet.all_text + ' ' +\
                                            tweet.quote_or_rt_text]
            if len(hashtags) == 1: #only aggregate single-hashtag tweets
                hashtag = hashtags[0].lower()
                if hashtag not in hashtag_dict:
                    hashtag_dict[hashtag] = tweet.all_text + ' ' +\
                                            tweet.quote_or_rt_text
                else:
                    hashtag_dict[hashtag] += ' ' + tweet.all_text + ' ' +\
                                              tweet.quote_or_rt_text

        result = list(hashtag_dict.values()) + tweets_without_hashtags

        return np.array(result)


    def _create_tweet_df(self):
        """ constructs a pandas dataframe that contains useful information
            about each tweet in the corpus
        """

        df = pd.DataFrame([{"date": x.created_at_datetime,
                            "text": x.all_text,
                            "user": x.screen_name,
                            "bio": x.bio,
                            "profile_location": x.profile_location,
                            "quote_or_rt_text": x.quote_or_rt_text,
                            "at_mentions": [u["screen_name"] for u in x.user_mentions],
                            "hashtags": x.hashtags,
                            "urls": x.most_unrolled_urls,
                            "geo": x.geo_coordinates,
                            "type": x.tweet_type,
                            "id": x.id} for x in self.tweet_object_list]).set_index("date")

        return df


if __name__ == '__main__':

    tc = TweetCorpus(['../data/03_28_2018_18_02.pkl',
                     '../data/03_30_2018_15_37.pkl'])
