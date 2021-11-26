#!/usr/local/bin/python3.9

import string
import re
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem.snowball import SnowballStemmer
from nltk import sent_tokenize, word_tokenize
import nltk
import pandas as pd

import twint

import nest_asyncio
nest_asyncio.apply()
c = twint.Config()

# extracting data from twitter

c.Search = "PS5"
c.Output = 'data/output.csv'
c.Lang = "en"
c.Pandas = True
c.Limit = 100

twint.run.Search(c)


def columne_names():
  return twint.output.panda.Tweets_df.columns


def twint_to_pd(columns):
  return twint.output.panda.Tweets_df[columns]


data = twint_to_pd(["tweet"])
data.head()
data["tweet"] = data["tweet"].str.replace("[^a-zA-Z0-9]", " ")
data["tweet"] = data["tweet"].apply(lambda x: " ".join([w for w in x.split() if len(w) > 3]) )

def tokenize(text):
        tokens=re.split("\W+", text)
        return tokens
data['tweet']=data['tweet'].apply(lambda x: tokenize(x.lower()))
nltk.download("stopwords")
stopwords = stopwords.words('english')
def remove_stopword(text):
    text_nostopword= [char for char in text if char not in stopwords]
    return text_nostopword

data['tweet']= data['tweet'].apply(lambda x: remove_stopword(x))
ps= nltk.PorterStemmer()
def stem(tweet_no_stopword):
       text = [ps.stem ( word) for word in tweet_no_stopword]
       return text
data["tweet"]= data["tweet"].apply(lambda x: stem(x))
data= pd.DataFrame(data["tweet"])
data_list = data.loc[:,"tweet"].to_list()
flat_data_list = [item for sublist in data_list for item in sublist]
data_count= pd.DataFrame(flat_data_list)
data_count= data_count[0].value_counts()
from nltk.probability import FreqDist
freq_count= FreqDist()
for words in data_count:
    freq_count[words] +=1
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
data_count = data_count[:20,]
plt.figure(figsize=(10,5))
sns.barplot(data_count.values, data_count.index, alpha=0.8)
plt.title('Top Words Overall')
plt.ylabel('Word from Tweet', fontsize=12)
plt.xlabel('Count of Words', fontsize=12)
plt.show()