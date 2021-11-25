#!/usr/local/bin/python3.9

# Author: Murtadha Marzouq
# Date:   12/10/2021
# Time:   12:00 PM
# Assignment: Social Media Analysis
# pip3 install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
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
import codecs
from numpy import add


# HERE I AM GOING TO PUT SWITCHES TO CHANGE THE OUTPUT
# Search Term
search_term = '#DeathPenalty'
# Make verbs in the root form
lemize_text = True
# Place any additional stop words here
additional_stop_words = ["amp", "https", "co", "rt", "new", "let",
                         "also", "still", "one", "people", "gt"]
# To calculate the sentiment of the tweets before preprocessing
get_sentiment_before_preprocessing = True


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

# Step 1: Fetching Tweets
# Setting up the Tweepy API to pull tweets
def run_twint():
    c = twint.Config()
    c.To = search_term
    c.Limit = 5
    c.Store_csv = True
    c.Output = 'data/output.csv'
    c.Pandas = True
    twint.run.Search(c)
    df = twint.storage.panda.Tweets_df
    #print(df['tweet'])
    
    return df

# setting up global variables


# Cleaning the tweets Step 2

def write_tweets_to_text_file(text_data):
    with open('text_data.txt', 'wb') as f:
        f.write(text_data, codecs.getwriter('utf-8')(f), ensure_ascii=False)


# Tokenization (2pt)


def tokenize(text):
    print('Tokenizing...')
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

    return lemmatized_text

# lemmatization (2pt) Method #2 TESTED


def lem(text):
    
    doc = nlp(u'{}'.format(text))
    tokens = []
    for token in doc:
        tokens.append(token)
    lemmatized_sentence = " ".join([token.lemma_ for token in doc])
    return lemmatized_sentence


# Cleaning the tweets Step 2
def clean_tweets_tb(input):
    punctuation = '!"$%&\'()*+,-./:;<=>?[\\]^_`{|}~•@'
    text = str(input)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub("@[A-Za-z0-9]+", "", text)
    text = re.sub(r"@[A-Za-z0-9]+", "", text)
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)
    text = re.sub(r"_[A-Za-z0-9]+", "", text)
    text = re.sub(r"__", "", text)
    text = re.sub(' +', ' ', text)
    text = re.sub('[' + punctuation + ']+', ' ', text)  # strip punctuation
    text = re.sub('\s+', ' ', text)  # remove double spacing
    text = re.sub('([0-9]+)', '', text)  # remove numbers
    text = "".join([char for char in text if char not in string.punctuation])
    text = text.lower()  # Lower text
    return text

# Applying PreProcessing


def apply_preprocessing(tweets):
    print('Applying Preprocessing...')
    # Setting up the stop words
    stop = nltk.corpus.stopwords.words('english')
    stop.extend(additional_stop_words)
    # cleaning the text of each tweet
    tweets['tweet'] = tweets['tweet'].apply(clean_tweets_tb)
    # split the tweets into words and remove words that are not in the stop list and less than 3 characters
    print('Removing Stop Words...')
    tweets['tweet'] = tweets['tweet'].apply(
        lambda x: " ".join(x for x in str(x).split() if x not in stop and len(x) > 3))

    # Apply lemmatization if the lemize_text is True
    if(lemize_text):
        print('Lemmatizing...')
        tweets['tweet'] = tweets['tweet'].apply(lambda x:   lem(x))
    tweets['cleaned_tweets'] = tweets['tweet']
    return tweets


# Sentiment Analysis Part 3
# Preparing the Data

    # Sentiment Analysis Part 3
    # This function is used to calculate the sentiment score of the tweets
    # -.1 or less if the tweet is negative
    # 0 if the tweet is neutral
    # .1 or more if the tweet is positive

def sentiment_analysis(tweet):
    print('Calculating Sentiment...')
    tweet['Polarity'] = (tweet['tweet'].map(
        lambda tweet: TextBlob(tweet).sentiment.polarity))
    
    tweet['Subjectivity'] = (tweet['tweet'].map(
        lambda tweet: TextBlob(tweet).sentiment.subjectivity))
    

    tweet["Sentiment"] = tweet["Polarity"].map(
        lambda pol: '+' if pol > 0 else '-')
    positive = tweet[tweet.Sentiment == "+"].count()["tweet"]
    negative = tweet[tweet.Sentiment == "-"].count()["tweet"]
    tweet["Sentiment"] = tweet["Polarity"].map(
        lambda pol: 'positive' if pol > 0 else 'negative' if pol < 0 else 'neutral')

    return tweet

# Ensure you check the switches on the top of the code to ensure that the correct data is being used


def main():
    # Applying PreProcessing
    df = run_twint()
    tweets = df
    # Getting the sentiment score of the tweets before preprocessing
    before_preprocessing_sentiment = sentiment_analysis(tweets)
    # print('Sentiment: ' + before_preprocessing_sentiment['Sentiment'].astype(str) + "  Polorization: " + before_preprocessing_sentiment['Polarity'].astype(str))  # Printing the sentiment score of the tweets before preprocessing
    print('Cleaning Tweets...')
    tweets = apply_preprocessing(tweets)
    # print(tweets['cleaned_tweets'])
    # # Sperator
    # print('\n')
    # print('\n')
    # Getting the sentiment score of the tweets after preprocessing
    after_preprocessing_sentiment = sentiment_analysis(tweets)
    # print('Sentiment: ' + after_preprocessing_sentiment['Sentiment'].astype(str) + " Polorization: " + after_preprocessing_sentiment['Polarity'].astype(str))  # Printing the sentiment score of the tweets before preprocessing

    df = tweets

    # Dumping the cleaned tweets to a CSV file
    print('Writing to CSV file...')
    if (get_sentiment_before_preprocessing):

        before_preprocessing_sentiment.to_csv('./data/Complete.csv')
        return after_preprocessing_sentiment
    else:
        after_preprocessing_sentiment.to_csv('./data/Complete.csv')
        return before_preprocessing_sentiment


# Running the main function
final_ouput = main()

# Showing the Polarity of the tweets using a search Term    

df_res_pandas = final_ouput
sns.distplot(df_res_pandas['Polarity'])

sns.set(rc={'figure.figsize':(11.7,8.27)})
plt.title('Distribution of Polarity of Tweets using a Search Term: ' + search_term)
#plt.show()

