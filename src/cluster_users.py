import pandas as pd
import numpy as np
from get_tweets import unpickle_results
from evaluate_nmf import umass_box_and_whiskers_for_diff_num_topics
from BuildNMF import BuildNMF


def load_tweets(pickle_file_names):
    """
    """
    all_tweets = []
    for fname in pickled_tweet_batches:
        all_tweets += unpickle_results(fname)

    return all_tweets


def get_bio_corpus(tweets):
    """
    """

    user_bios =  {t.user_id: t.bio for t in tweets if t.bio}

    ids = []
    bio_corpus = []
    for user_id, bio in user_bios.items():
        ids.append(user_id)
        bio_corpus.append(bio)
    bio_corpus = np.array(bio_corpus)

    return bio_corpus


if __name__ == '__main__':

    pickled_tweet_batches = ['../data/03_28_2018_18_02.pkl',
                             '../data/03_30_2018_15_37.pkl']

    tweets = load_tweets(pickled_tweet_batches)

    corpus = get_bio_corpus(tweets)


    # Generate boxplots for topic coherences of bios
    # num_topics_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # umass_box_and_whiskers_for_diff_num_topics(corpus, num_topics_list,
    #                 max_iter=200, savepath='../plots/boxplots_bios.png',
    #                                                show=True, M=5)

    nmf_mod = BuildNMF(corpus, num_topics=50)
    nmf = nmf_mod.fit()
