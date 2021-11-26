#!/usr/local/bin/python3.9

# Author: Murtadha Marzouq
# Date:   12/10/2021
# Time:   12:00 PM
# Assignment: Word Frequency Analysis
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

# TEXT PROCESSING libs
import re
from textblob import TextBlob
import logging
import gensim
from gensim import corpora, models, similarities
import tempfile
from sklearn.feature_extraction.text import CountVectorizer
import gensim as gensimvis
import twint
from gensim.models.phrases import Phrases, Phraser
from gensim.models import Word2Vec
from gensim.test.utils import get_tmpfile
from gensim.models import KeyedVectors




# This function  converts a column of words into a list of words
def column_to_string(df, column_name):
    return df[column_name].astype(str).str.cat(sep=' ')



def drop_columns(table):
    table = table.drop(["Unnamed: 0",'id','timezone', 'place','language', 'hashtags',
       'cashtags', 'user_id', 'username', 'name', 'day', 'hour', 'nlikes',
       'search','conversation_id', 'created_at', 'user_id_str', 'link', 'urls', 'photos', 'video',
       'thumbnail', 'retweet','nreplies', 'nretweets', 'quote_url', 'near', 'geo', 'source', 'user_rt_id', 'user_rt',
       'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src',
       'trans_dest'],axis = 1)
    return table


def plot_sentiment(table):
    pal = {"positive":'r', "negative":"g","neutral":"b"}
    fig1 = sns.displot(table, x="Sentiment", hue="Sentiment", legend=False, palette= pal)
    fig1.fig.suptitle("Count of tweets by Sentiment",fontsize =15)
    plt.tight_layout()
    plt.show()






# Reading From File
# tweets = pd.read_csv('data/Completed100.csv')

# # Converting the column to string
# text = column_to_string(tweets, 'cleaned_tweets').split(' '
#                                                         )
# # Calculating the frequency of each word

# # Printing the top 10 words
# print(word_frequency[:10])

# visualizing the frequency of each word


def visualize_word_frequcency(table):
    word_frequency = table.apply(' '.join(table['cleaned_tweets']).split()).value_counts()
    word_count = word_frequency
    word_count = word_count[:10, ]
    plt.figure(figsize=(10, 5))
    sns.barplot(word_count.index, word_count.values, alpha=0.8)
    plt.title('Top 10 Words in Tweets')
    plt.ylabel('Frequency', fontsize=14)
    plt.xlabel('Term', fontsize=14)
    plt.show()



tweets = pd.read_csv('data/Completed100.csv')
tweets = drop_columns(tweets)
plot_sentiment(tweets)
visualize_word_frequcency(tweets)
print(tweets)