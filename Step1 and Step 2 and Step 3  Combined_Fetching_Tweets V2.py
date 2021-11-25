#!/usr/local/bin/python3.9

# Author: Murtadha Marzouq
# Date:   12/10/2021
# Time:   12:00 PM
# Assignment: Social Media Analysis
# pip3 install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
import codecs
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import os
import textblob
from textblob import TextBlob, Word
import sklearn
import pandas as pd
import matplotlib.pyplot as plt
import twint
import string
import spacy
from textblob import TextBlob
import nltk
from nltk.tokenize import word_tokenize
import nest_asyncio
nest_asyncio.apply()
nltk.download('wordnet')
wnl = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
os.getcwd()
nlp = spacy.load('en_core_web_sm')


# Setting up the Tweepy API to pull tweets
def run_twint():
    #
    c = twint.Config()
    c.To = '#DeathPenalty'
    c.Limit = 5
    c.Store_csv = True
    c.Output = 'data/output.csv'
    c.Pandas = True
    twint.run.Search(c)
    c.Limit = 1
    df = twint.storage.panda.Tweets_df
    # print(df)
    df.head()

    # Step 1: Fetching Tweets
    tweets = df[['tweet']]
    return df

# setting up global variables


# Cleaning the tweets Step 2

def write_tweets_to_text_file(text_data):
    with open('text_data.txt', 'wb') as f:
        f.write(text_data, codecs.getwriter('utf-8')(f), ensure_ascii=False)


# Tokenization (2pt)


def tokenize(text):
    tok = nltk.toktok.ToktokTokenizer()
    tokens = tok.tokenize(text)
    print('No of tokens:', len(tokens))
    return tokens


# Lemmatization (2pt)
def pos_tag_wordnet(tagged_tokens):
    tag_map = {'j': wordnet.ADJ, 'v': wordnet.VERB,
               'n': wordnet.NOUN, 'r': wordnet.ADV}
    new_tagged_tokens = [(word, tag_map.get(tag[0].lower(), wordnet.NOUN))
                         for word, tag in tagged_tokens]
    return new_tagged_tokens


# lemmatization (2pt) Method #1
def lemmatize_text(input):
    tagged_tokens = nltk.pos_tag(
        nltk.word_tokenize(input))  # Positonal tagging
    wordnet_tokens = pos_tag_wordnet(tagged_tokens)
    lemmatized_text = [wnl.lemmatize(word, pos)
                       for word, pos in wordnet_tokens]

    print(lemmatized_text)
    return lemmatized_text

# lemmatization (2pt) Method #2 TESTED


def lem(text):
    # print('Before Lems: '+ text)
    doc = nlp(u'{}'.format(text))
    tokens = []
    for token in doc:
        tokens.append(token)
    lemmatized_sentence = " ".join([token.lemma_ for token in doc])
    # print('After Lems: '+ lemmatized_sentence)

    print(lemmatized_sentence)


# Cleaning the tweets Step 2
def clean_tweets_tb(input):
    text = str(input)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub("@[A-Za-z0-9]+", "", text)
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)
    text = re.sub(r"_[A-Za-z0-9]+", "", text)
    text = re.sub(r"__", "", text)
    text = re.sub(' +', ' ', text)
    text = "".join([char for char in text if char not in string.punctuation])
    text = text.lower()  # Lower text
    return text

# Applying PreProcessing


def apply_preprocessing(tweets):
    tweets['tweet_tb'] = tweets['tweet'].apply(clean_tweets_tb)
    tweets['cleaned_tweets'] = tweets['tweet'].apply(clean_tweets_tb)

    stop = nltk.corpus.stopwords.words('english')
    stop.extend(["amp", "https", "co", "rt", "new", "let",
                "also", "still", "one", "people", "gt"])
    tweets['cleaned_tweets'] = tweets['tweet_tb'].apply(
        lambda x: " ".join(x for x in str(x).split() if x not in stop))
    tweets['cleaned_tweets'] = tweets['cleaned_tweets'].apply(lambda x:   lem(x))
    print(tweets['tweet_tb'])
    return tweets


# Sentiment Analysis Part 3
# Preparing the Data

    # Sentiment Analysis Part 3
    # This function is used to calculate the sentiment score of the tweets
    # -.1 or less if the tweet is negative
    # 0 if the tweet is neutral
    # .1 or more if the tweet is positive

def sentiment_analysis(tweet):
    tweet['Polarity'] = (tweet['tweet'].map(
        lambda tweet: TextBlob(tweet).sentiment.polarity))
    tweet["Sentiment"] = tweet["Polarity"].map(
        lambda pol: '+' if pol > 0 else '-')
    positive = tweet[tweet.Sentiment == "+"].count()["tweet"]
    negative = tweet[tweet.Sentiment == "-"].count()["tweet"]
    tweet["Sentiment"] = tweet["Polarity"].map(
        lambda pol: 'positive' if pol > 0 else 'negative' if pol < 0 else 'neutral')
   
    return tweet


def main():
    # Applying PreProcessing
    df = run_twint()
    tweets = df[['tweet']]
    tweets = apply_preprocessing(tweets)
    df['cleaned_tweets'] = tweets['tweet_tb']
    # Dumping the cleaned tweets to a CSV file
    return sentiment_analysis(df)


final_ouput = main()
# Writing The  Combined File
final_ouput.to_csv('./data/Complete.csv')