from tweet_parser.tweet import Tweet
from searchtweets import (ResultStream,
                           collect_results,
                           gen_rule_payload,
                           load_credentials)
import pickle

def get_search_args(search_option='tweets'):
    '''
    input:
    - search_option: either 'tweets' or 'counts' depending on desired query
    returns:
    - search_args: argument of get_tweets function used to collect
    tweets or counts of tweets
    '''

    search_args = load_credentials(filename="~/.twitter_keys.yaml",
                                   account_type="premium",
                                   yaml_key='search_tweets_api_'+search_option)
    return search_args


def get_tweets(num_tweets, search_rule, results_per_call=500,
               search_option='tweets', from_date=None, to_date=None):
    '''
    input:
    - num_tweets: approximate number of tweets to collect
    - search_rule: string representing search query (see link below for details)
    https://developer.twitter.com/en/docs/tweets/search/overview/premium
    - results_per_call: how many results each API call will return (up to 500)
    - search_option: either 'tweets' or 'counts' depending on desired query
    - from_date/to_date: specify the time period within which to look for
    tweets (format = 'YYYY-MM-DD')
    returns:
    - list of tweets (or counts) collected
    '''
    tweets = []
    search_args = get_search_args(search_option)

    rule_a = gen_rule_payload(search_rule, from_date=from_date,
                             to_date=to_date, results_per_call=results_per_call)

    if num_tweets > 500:
        return

    tweets = collect_results(rule_a, max_results=num_tweets,
                    result_stream_args=search_args)


    #i = 0
    # while i < num_tweets():
    #     tweets += collect_results(rule_a, max_results=500,
    #                               result_stream_args=search_args)
    #     i+=results_per_call
    ## need to change page number after each call? if so, how?

    return tweets


def pickle_results(results_list, fname):
    with open(fname, 'wb') as f:
        pickle.dump(results_list,f)

def unpickle_results(fname):
    with open(fname, 'rb') as f:
        recovered_results_from_pkl = pickle.load(f)

    return recovered_results_from_pkl

if __name__ == '__main__':

    #eventually make this accept command line args?
    search_rule = """
                  ("changed my mind" OR
                  "changed my opinion")
                  -"never changed"
                  -"haven't changed"
                  -"hasn't changed"
                  -"have not changed"
                  -"has not changed"
                  -is:retweet
                  """
    collected_tweets = get_tweets(500, search_rule)
    pickle_results(collected_tweets, 'query_results_03_26_2018_1.pkl')
    recovered_results_from_pkl = unpickle_results('query_results_03_26_2018_1.pkl')
