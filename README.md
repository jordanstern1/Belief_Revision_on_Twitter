# Studying Belief Revision on Twitter

## Introduction and Project Goals

The goal of this project is to investigate cases wherein people overtly express a change of opinion on a subject via Twitter. More specifically, the aim is to attempt to answer at least the first of the following questions: (1) With respect to what issues are people most likely to change their opinion? (2) What kinds of people are changing their minds? (3) What rhetorical strategies make people more likely to change their minds?

To obtain the necessary data to conduct this analysis, I have used one of Twitter's APIs to query for tweets containing the phrases "changed my mind," "changed my opinion," or "changed my view." To be more certain that the resulting tweets actually correspond to a change of opinion, I excluded negations of these statements (e.g., "haven't/hasn't changed my mind") and I also excluded retweets (since retweets do not necessarily represent endorsements of the original tweet).

 I have begun to address the first question in the opening paragraph, however much work remains to be done. I expect that people are more likely to change their minds on relatively trivial issues—particularly those issues that are not heavily politicized—but it will be interesting to test this hypothesis.

 After providing a sufficient answer to the first question, I plan to investigate the second question posed in the opening paragraph by examining biographical data regarding the authors of each tweet in the corpus. It’s possible that younger Twitter users are more likely to change their minds, or that Twitter users located in urban areas are more likely to change their minds.

In order to address the third question posed in the opening paragraph, further investigation may include an analysis of the content of the news articles that are linked to in the tweet corpus. Specifically, I would pull links to articles included in the tweet corpus and web scrape in order to obtain the content of these articles. Next, I would create a TF-IDF matrix from this corpus of articles and possibly use K-means clustering to group articles based on the similarity of their textual content. I would then investigate some of the following questions: are these articles relatively short or long? Do these articles tend to come from reputable organizations? Are there any interesting commonly occurring words across many of these articles?


## Data Collection and Analysis

Using Twitter's premium API I collected 55,539 unique tweets (authored between January 26, 2018 and March 28, 2018) that matched my search query. Next I assembled a corpus containing the text content from each of the tweets, then I applied standard natural language processing methods to clean this corpus (lowercase, remove punctuation, and stem/lemmatize). Next, I computed a TF-IDF matrix and began topic modeling via Non-negative Matrix Factorization (NMF) to determine the domains within which people are most commonly changing their minds (this addresses the first question in the opening paragraph).

The main challenge of topic modeling given a corpus of tweets is that each tweet
contains relatively little content, whereas more traditional applications of topic modeling use longer documents such as news articles. Consequently there are relatively few co-occurrences of substantive words across tweets, which makes it difficult to
establish coherent topics. Recent work in the field of topic modeling with Twitter data has established several methods for dealing with this issue. For example, Steinskog et al. have recommended aggregation of tweets by hashtag and by author
in order to synthesize longer documents that result in increased topic
coherence [1].

Before employing the aggregation method, I have assessed the coherence of my
topics using the UMass coherence score recommended by Steinskog et al. [1]. As an example the bar chart below shows UMass scores for each topic when there were five total topics (Note: the UMass score tends to zero as a topic becomes increasingly coherent).

![Alt text](plots/coherence_scores1.png)

## Preliminary Results

From my initial implementations of NMF, a couple of fairly coherent latent topics have emerged.

1\. *Twitter users expressing a change of opinion on issues surrounding immigration*

Top 10 most common (stemmed) words within topic:
  * 'blue'
  * 'kid'
  * 'daca'
  * 'amnesti'
  * 'state'
  * 'breitbart'
  * 'via'
  * 'whi'
  * '#daca'
  * 'legal'


2\. *Twitter users expressing a change of opinion on issues surrounding Brexit*

Top 10 most common (stemmed) words within topic:
  * 'thi'
  * 'tweet'
  * 'ha'
  * 'referendum'
  * 'like'
  * 'vote'
  * 'one'
  * 'year'
  * 'brexit'
  * 'need'


3\. *Twitter users expressing a change of opinion on issues related to entertainment products*

Top 10 most common (stemmed) words within topic:
  * 'one'
  * 'xbox'
  * 'tabl'
  * 'turn'
  * 'video'
  * 'playlist'
  * 'ad'
  * 'buy'
  * 'onli'
  * 'day'



## Citations
[1] [Twitter Topic Modeling By Tweet Aggregation](http://www.aclweb.org/anthology/W17-0210)
