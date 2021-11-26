#!/usr/local/bin/python3.9

# Author: Murtadha Marzouq
# Date:   12/10/2021
# Time:   12:00 PM
# Assignment: Social Media Analysis
# pip3 install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
import datetime
import nest_asyncio
from nltk.tokenize import word_tokenize
import nltk
from textblob import TextBlob
import spacy
import string
import seaborn as sns
import twint
import matplotlib.pyplot as plt
import pandas as pd
import sklearn
from textblob import TextBlob, Word
import textblob
import os
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet
import pyLDAvis
from numpy import add
import gensim
from gensim import models
from gensim.models.coherencemodel import CoherenceModel
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import gensim
import re
import nltk
from tqdm import tqdm
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.manifold import TSNE
from datetime import date
import datetime
import twint





nlp = spacy.load('en_core_web_sm')
tokens = []

def lem(text):
    
    doc = nlp(u'{}'.format(text))
    for token in doc:
        tokens.append(token)
    lemmatized_sentence = " ".join([token.lemma_ for token in doc])
    return lemmatized_sentence


tweets = pd.read_csv('./data/Complete.csv')
text = tweets['cleaned_tweets'].apply(lem)
print(text)
print(tokens)
bigrams = text.to_frame()

tweet_token_list = [text[i]+'_'+text[i+1] for i in range(len(text)-1)]
print(tweet_token_list)
print((bigrams.iloc[0,0]))
from sklearn.feature_extraction.text import CountVectorizer

# the vectorizer object will be used to transform text to vector form
vectorizer = CountVectorizer()

# apply transformation
tf = vectorizer.fit_transform(bigrams['cleaned_tweets']).toarray()

# tf_feature_names tells us what word each column in the matric represents
bigram_names = vectorizer.get_feature_names()


tf = pd.DataFrame(tf)
tf.columns = [bigram_names]
sample_tf = tf.head()
tf.loc['total',:] = tf.sum(axis=0)
print(tf.loc['total'])
topicslist = pd.DataFrame(data = tf.loc['total'])
topicslist = topicslist.reset_index()
topicslist = topicslist.rename(columns = {"level_0":"bigrams"})
topicslist = topicslist.sort_values(by = ['total'], ascending = False)
topicslist_10 = topicslist[:10]

topicslist_10['bigrams']
fig = plt.figure()
fig.set_figwidth(15)
plt.bar(topicslist_10['bigrams'], topicslist_10['total'])
plt.xlabel('Topic')
plt.ylabel('Total Popularity')
print(topicslist_10)

plt.show()
