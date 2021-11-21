from transformers import pipeline
from transformers import BertModel, BertTokenizer, AdamW, get_linear_schedule_with_warmup
import tensorflow_datasets as tfds
import torch
import numpy as np

import pandas as pd
import tensorflow.compat.v2 as tf  
import seaborn as sns
import matplotlib.pyplot as plt
from pylab import rcParams
sns.set(style='whitegrid', palette='muted', font_scale=1)

HAPPY_COLORS_PALETTE = ["#01BEFE", "#FFDD00", "#FF7D00", "#FF006D", "#ADFF02", "#8F00FF"]
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline
sns.set_palette(sns.color_palette(HAPPY_COLORS_PALETTE))
ds = tfds.load(twint.output.panda.Tweets_df)
assert isinstance(ds, tf.data.Dataset)

def to_sentiment(rating):
    
  rating = int(rating)

  if rating <= 2:

    return 0

  elif rating == 3:

    return 1

  else:

    return 2


import twint
# Set up TWINT config
c = twint.Config()
c.Search = "#death"
# Custom output format
c.Limit = 1
c.Pandas = True
c.Store_csv = True
c.Output = "tweets.csv"
c.Pretty = True

twint.run.Search(c)

def column_names():
    return twint.storage.panda.Tweets_df.columns
def twint_to_pd(columns):
    return twint.storage.panda.Tweets_df[columns]

column_names()
tweet_df = twint_to_pd(column_names())
print(len(tweet_df))
nlp = pipeline('sentiment-analysis')

df = twint.output.panda.Tweets_df[column_names()]
df['sentiment'] = 1

df['sentiment'] = df['sentiment'].apply(to_sentiment)

df['sentiment_str'] = df['sentiment'].map({0: 'Negative', 1: 'Neutral', 2: 'Positive'})

df['sentiment_color'] = df['sentiment'].map({0: 'red', 1: 'blue', 2: 'green'})















sns.countplot(data=df, palette=HAPPY_COLORS_PALETTE)

plt.xlabel('tweet');
plt.ylabel('count');
plt.title('sentiment');
plt.show()
