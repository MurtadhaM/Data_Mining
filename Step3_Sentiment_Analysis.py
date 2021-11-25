#!/usr/local/bin/python3.9

import pandas as pd
import numpy as np
from textblob import TextBlob , classifiers
import matplotlib.pyplot as plt



# #### Step 3: Sentiment Analysis (8pts total) ####

def sentiment_analysis(tweet):
  


     #for each_tweet in input_data.tweet:
     #   sentiment_analysis(each_tweet)
    tweet['Polarity'] = (tweet['tweet'].map(lambda tweet: TextBlob(tweet).sentiment.polarity))
    tweet["Sentiment"] = tweet["Polarity"].map(lambda pol: '+' if pol > 0 else '-')
    positive = tweet[tweet.Sentiment == "+"].count()["tweet"]
    negative = tweet[tweet.Sentiment == "-"].count()["tweet"]
    tweet["Sentiment"] = tweet["Polarity"].map(lambda pol: 'positive' if pol > 0  else 'negative' if pol < 0      else 'neutral') 
    tweet.to_csv('date/tweet_sentiment.csv')
    return tweet


# Setting up a TextBlob object
input_data = pd.read_csv('data/Completed100.csv')
#textblob = pd.DataFrame(input_data)
sentiment_analysis(input_data)

































