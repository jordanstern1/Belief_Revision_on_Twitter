import pandas as pd
import numpy as np
from get_tweets import unpickle_results
from evaluate_nmf import umass_box_and_whiskers_for_diff_num_topics
from BuildNMF import BuildNMF
from make_wordcloud import *
from evaluate_nmf import *


def load_tweets(pickle_file_names):
    """ load tweets from pickle files

    input:
    - pickle_file_names (list of strings):
        list of names of .pkl files storing tweet objects

    returns:
    - list of tweet objects

    """
    all_tweets = []
    for fname in pickled_tweet_batches:
        all_tweets += unpickle_results(fname)

    return all_tweets


def get_bio_corpus(tweets):
    """ given a list of tweet objects, assemble a corpos of Twitter user bios

    input:
    - tweets (list or np array):
        list of tweet objects from .pkl files

    returns:
    - list of user IDs
    - corpus of Twitter bios corresponding to each user ID

    """

    user_bios =  {t.user_id: t.bio for t in tweets if t.bio and len(t.bio)>20}

    ids = []
    bio_corpus = []
    for user_id, bio in user_bios.items():
        ids.append(user_id)
        bio_corpus.append(bio)
    bio_corpus = np.array(bio_corpus)

    return ids, bio_corpus


if __name__ == '__main__':

    pickled_tweet_batches = ['../data/03_28_2018_18_02.pkl',
                             '../data/03_30_2018_15_37.pkl']

    tweets = load_tweets(pickled_tweet_batches)

    ids, corpus = get_bio_corpus(tweets)


    # Generate boxplots for topic coherences of bios
    # num_topics_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
    # umass_box_and_whiskers_for_diff_num_topics(corpus, num_topics_list,
    #      max_iter=200, savepath='../plots/coherence_score_boxplots_bios.png',
    #                                                show=True, M=5)


    # Get wordclouds for all topics for corpus of twitter user bios(50 topics)
    ideal_num_topics = 10 # determined from coherence score boxplots
    nmf_mod = BuildNMF(corpus, num_topics=ideal_num_topics)
    nmf = nmf_mod.fit(display=True)
    # gen_wordclouds_for_all_topics(nmf_mod)

    # Plot bar chart of topic sizes
    get_bar_chart_of_topic_size(nmf_mod, 'Sizes of Twitter Bio Topics',
                                'Number of Bios in Topic',
                                savepath='../plots/bio_topic_sizes.png')
