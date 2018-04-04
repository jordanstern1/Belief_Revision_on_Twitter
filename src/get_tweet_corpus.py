from get_tweets import unpickle_results
import pandas as pd
import numpy as np


class TweetCorpus(object):
    """
    Given a list of .pkl files containing lists of tweet objects, this class
    will recover the tweet objects from the .pkl files and assemble various
    corpora of tweets (stored as class attributes) that can then be used for
    topic modeling using the BuildNMF class.

    Methods
    --------
    -


    Attributes
    -----------
    -

    """
    def __init__(self, pickled_tweet_batches):
        self.pickled_tweet_batches = pickled_tweet_batches
        self.tweet_object_list = self._get_corpus('objects')

        self.raw_tweet_corpus = self._get_corpus('raw_text')
        self.quote_aggregated_corpus = self._get_corpus('aggregate_quote_tweets')
        self.hashtag_aggregated_corpus = self._get_corpus('aggregate_by_hashtag')
        self.tweet_df = self._create_tweet_df()


    def _get_corpus(self, option):
        """ recover tweet raw tweet objects from .pkl files

        Note: When we have a quote tweet (tweet that is a response to a quoted
        tweet), we combine the text of the quote tweet and the quoted tweet.

        input:
         - list of pickle file names containing tweet batches collected using
         get_tweets.py script

        returns:
        - list of tweet objects
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
        """ aggregate tweets by hashtag """

        hashtag_dict = {}
        tweets_without_hashtags = []
        for tweet in self.tweet_object_list:
            if len(tweet.hashtags) == 0:
                tweets_without_hashtags += [tweet.all_text + ' ' +\
                                            tweet.quote_or_rt_text]
            for hashtag in tweet.hashtags:
                hashtag = hashtag.lower()
                if hashtag not in hashtag_dict:
                    hashtag_dict[hashtag] = tweet.all_text + ' ' +\
                                            tweet.quote_or_rt_text
                else:
                    hashtag_dict[hashtag] += ' ' + tweet.all_text + ' ' +\
                                              tweet.quote_or_rt_text

        result = list(hashtag_dict.values()) + tweets_without_hashtags

        return list


    def _create_tweet_df(self):
        """ Create a pandas dataframe """

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
                            "id": x.id} for x in all_tweets]).set_index("date")

        return df

if __name__ == '__main__':
    # all_tweets = load_tweet_objects(['../data/03_28_2018_18_02.pkl',
    #                                  '../data/03_30_2018_15_37.pkl'])

    tc = TweetCorpus(['../data/03_28_2018_18_02.pkl',
                     '../data/03_30_2018_15_37.pkl'])
