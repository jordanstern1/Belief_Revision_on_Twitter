from tweet_parser.tweet import Tweet
from searchtweets import (ResultStream,
                           collect_results,
                           gen_rule_payload,
                           load_credentials)
import pickle
import datetime


def get_search_args(search_option='tweets'):
    '''
    ---------------------------------------------------------------------------
    input:
    - search_option: either 'tweets' or 'counts' depending on desired query
    ---------------------------------------------------------------------------
    returns:
    - search_args: argument of get_tweets function used to collect
    tweets or counts of tweets
    ---------------------------------------------------------------------------
    '''

    search_args = load_credentials(filename="~/.twitter_keys.yaml",
                                   account_type="premium",
                                   yaml_key='search_tweets_api_'+search_option)

    return search_args


def get_tweets(num_tweets, search_rule, results_per_call=500,
               search_option='tweets', from_date=None, to_date=None,
               count_bucket=None):
    '''
    ---------------------------------------------------------------------------
    input:
    - num_tweets: approximate number of tweets to collect
    - search_rule: string representing search query (see link below for details)
    https://developer.twitter.com/en/docs/tweets/search/overview/premium
    - results_per_call: how many results each API call will return (up to 500)
    - search_option: either 'tweets' or 'counts' depending on desired query
    - from_date/to_date: specify the time period within which to look for
    tweets (format = 'YYYY-MM-DD')
    ---------------------------------------------------------------------------
    output:
    - .pkl file storing results of query.
    - Name of output .pkl file includes the date and time immediately after the
    query was executed (path = 'data/mm_dd_yyyy_HH_MM.pkl')
    ---------------------------------------------------------------------------
    returns:
    - list of tweets (or counts) collected
    ---------------------------------------------------------------------------
    '''
    tweets = []
    search_args = get_search_args(search_option)

    rule_a = gen_rule_payload(search_rule, from_date=from_date,
                             to_date=to_date, results_per_call=results_per_call,
                             count_bucket=count_bucket)

    print("\nHere's your rule:\n" + rule_a + '\n')


    # double check to prevent user from accidentally exhausting API calls
    x = input('Are you sure you want to proceed? (y/n) ').lower()
    if x in ['n', 'no']:
        print('Query cancelled.')
        return
    elif x in ['y', 'yes']:
        print('Collecting tweets...')
        tweets = collect_results(rule_a, max_results=num_tweets,
                        result_stream_args=search_args)
        pickle_results(tweets)

    #i = 0
    # while i < num_tweets():
    #     tweets += collect_results(rule_a, max_results=500,
    #                               result_stream_args=search_args)
    #     i+=results_per_call
    ## need to change page number after each call? if so, how?

    return tweets


def pickle_results(results_list):
    '''
    ---------------------------------------------------------------------------
    input:
    - results_list: list of collected tweets/counts returned by query
    ---------------------------------------------------------------------------
    output:
    - .pkl file storing results of query.
    - Name of output .pkl file includes the date and time immediately after the
    query was executed (path = 'data/mm_dd_yyyy_HH_MM.pkl')
    ---------------------------------------------------------------------------
    returns:
    - None
    ---------------------------------------------------------------------------
    '''

    now = datetime.datetime.now()
    fname = 'data/' + str(now.strftime("%m_%d_%Y_%H_%M")) + '.pkl'

    with open(fname, 'wb') as f:
        pickle.dump(results_list,f)

    print('\nResults saved at this location:', fname)


def unpickle_results(fname):
    '''
    ---------------------------------------------------------------------------
    input:
    - fname: path of file to be unpickled
    ---------------------------------------------------------------------------
    returns:
    - data stored in .pkl file (query results)
    ---------------------------------------------------------------------------
    '''

    with open(fname, 'rb') as f:
        recovered_results_from_pkl = pickle.load(f)

    return recovered_results_from_pkl


if __name__ == '__main__':

    search_rule = """
                  ("changed my mind" OR
                  "changed my opinion" OR
                  "changed my view")
                  -"not changed"
                  -"never changed"
                  -"haven't changed"
                  -"hasn't changed"
                  -"may have changed"
                  -"might have changed"
                  -"might've changed"
                  -"has ever changed"
                  -"jk"
                  -"just kidding"
                  -is:retweet
                  """

    collected_tweets = get_tweets(500, search_rule, search_option='counts',
                                  count_bucket='day')
