from get_tweet_corpus import TweetCorpus
import numpy as np
import pandas as pd
import csv
from statsmodels.stats.proportion import proportion_confint


def get_random_sample(tc, fpath, num):
    """ We ultimatley want to create a markdown table with sample of 500 tweets
        and whether or not each tweet is a meaningful indication of a Twitter
        expressing a change of opinion. This function grabs a random sample of
        num tweets from the corpus and writes them to a csv file in a format
        that can be directly converted to markdown using this
        link: https://donatstudios.com/CsvToMarkdownTable

    input:
    - tc (TweetCorpus object):
        has 'raw_tweet_corpus' attribute from which we will randomly sample
    - fpath (string):
        file path of output .csv file
    - num (int):
        size of random sample

    output:
    - .csv file later to be converted to a markdown table

    returns:
    - list of tweets (strings) in random sample

    """

    random_sample = list(np.random.choice(tc.raw_tweet_corpus, num))

    with open(fpath, 'w', encoding='utf-8-sig') as f:
        wr = csv.writer(f)
        wr.writerow(['Index', 'Tweet Content', 'Meaningful? (Y/N)'])
        for idx, item in enumerate(random_sample):
            item = item.replace('"', '')
            item = item.replace("'", '')
            item = item.replace(",", '')
            item = item.replace("\n", '')
            item =  item.replace("|", '')
            wr.writerow([idx+1, item])

    return random_sample


def get_conf_intvl(fname):
    """ get a confidence interval on the proportion of tweets that are not
        meaningful indications of someone expressing a change of opinion

    input:
    - fname (string):
        name of csv file outputted by get_random_sample() and edited to include
        my reading of whether or not ('Y' or 'N') a tweet is a meaningful
        indication of someone expressing a change of opinion

    returns:
    - df (pandas dataframe):
        shows my label for each tweet in the sample ('Y' or 'N')
    - ci_low (float):
        lower and bound of confidence interval for the proportion of
        tweets that are not meaningful
    - ci_upp (float):
        upper bound of confidence interval for the proportion of
        tweets that are not meaningful
        
    """

    df = pd.read_csv(fname)
    num_successes = df[df['Meaningful? (Y/N)']=='N'].shape[0]
    num_observations = df['Meaningful? (Y/N)'].size

    ci_low, ci_upp = proportion_confint(num_successes, num_observations,
                                        alpha=0.05, method='agresti_coull')

    return df, ci_low, ci_upp


if __name__ == '__main__':

    # tc = TweetCorpus(['../data/03_28_2018_18_02.pkl',
    #                   '../data/03_30_2018_15_37.pkl'])

    # get random sample of tweets and write them to a csv file
    #sample = get_random_sample(tc, '../data/random_tweets.csv', 500)

    # get confidence interval on proportion of tweets that are not meaningful
    # NOTE: I labelled 200 tweets as meaningful ('Y') or not meaningful ('N')
    df, ci_low, ci_upp = get_conf_intvl('../data/random_tweets_200_labelled.csv')
    print("CI for proportion of tweets that are not meaningful: " +\
          "[{:.2f}, {:.2f}]".format(ci_low, ci_upp))
