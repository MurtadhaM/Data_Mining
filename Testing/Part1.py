#!/usr/local/bin/python3.9

from matplotlib import pyplot as plt
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk import tokenize
import pandas as pd
import seaborn as sns

import numpy as np

import re

import string

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.model_selection import train_test_split

from nltk.stem import PorterStemmer

from nltk.stem import WordNetLemmatizer

# ML Libraries

from sklearn.metrics import accuracy_score

from sklearn.naive_bayes import MultinomialNB

from sklearn.linear_model import LogisticRegression

from sklearn.svm import SVC

import pandas as pd
# Global Parameters

stop_words = set(stopwords.words('english'))


def get_feature_vector(train_fit):

    vector = TfidfVectorizer(sublinear_tf=True)

    vector.fit(train_fit)

    return vector


data = pd.read_csv('/Users/m/Documents/GitHub/Data_Mining/data/Complete.csv')
df = pd.DataFrame(data)
ct = df['cleaned_tweets']

# df.head()

print(df['cleaned_tweets'])
# print(data)

text = data.tweet


# the vectorizer object will be used to transform text to vector form
# make a new column with only the popular hashtags
vectorizer = CountVectorizer(
    max_df=0.9, min_df=12, token_pattern='\w+|\$[\d\.]+|\S+')
tf = vectorizer.fit_transform(df['cleaned_tweets']).toarray()
print(tf)
tf_feature_names = vectorizer.get_feature_names()

# apply transformation

#tf = vectorizer.fit_transform(df['tweet']).toarray
# tf_feature_names tells us what word each column in the matric represents
tf_feature_names = vectorizer.get_feature_names()


number_of_topics = 9
print(tf_feature_names)
model = LatentDirichletAllocation(
    n_components=number_of_topics, random_state=0)
model.fit(tf)

topic_dict = {}

def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        topic_dict["Topic %d words" % (topic_idx)] = ['{}'.format(feature_names[i])
                                                      for i in topic.argsort()[:-no_top_words - 1:-1]]
        topic_dict["Topic %d weights" % (topic_idx)] = ['{:.1f}'.format(topic[i])
                                                        for i in topic.argsort()[:-no_top_words - 1:-1]]
    return pd.DataFrame(topic_dict)


no_top_words = 10

test = display_topics(model, tf_feature_names, no_top_words)

plt.figure(figsize=(10, 5))
sns.barplot(y=tf, alpha=0.8)
plt.title('Top 10 Words in Tweets')
plt.ylabel('Frequency', fontsize=14)
plt.xlabel('Term', fontsize=14)
plt.show()

print(test)
