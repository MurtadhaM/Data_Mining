#!/usr/local/bin/python3.9

# Author: Murtadha Marzouq
# Date:   12/10/2021
# Time:   12:00 PM
# Assignment: Social Media Analysis
# pip3 install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint

from datetime import timedelta
import datetime
import nest_asyncio #Package used to prevent a runtime error in Jupyter Notebooks.
from nltk.probability import FreqDist
from nltk.tokenize import word_tokenize
import nltk
from textblob import TextBlob
import spacy
import string
import seaborn as sns #Visualization tool used.
import twint #Package used to collect Twitter data without a Twitter developer account.
import matplotlib.pyplot as plt
import pandas as pd #Used to put data into dataframe.
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
search_term = '#Police'
# Make verbs in the root form
lemize_text = True
# Place any additional stop words here
#Choose how many max tweets to return
limit = 50
additional_stop_words = ["amp", "https", "co", "rt", "new", "let",
                         "also", "still", "one", "people", "gt"]
# To calculate the sentiment of the tweets before preprocessing
get_sentiment_before_preprocessing = True

# Start start 
since_in_day = 5

nest_asyncio.apply()
nltk.download('wordnet')
wnl = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
os.getcwd()
nlp = spacy.load('en_core_web_sm') #Likely point of issues when first running this program.

# Step 1: Fetching Tweets
# Setting up the Tweepy API to pull tweets
def run_twint():
    #This is used to start pulling Tweets from the most recent time.
    since = (datetime.datetime.now() - timedelta(since_in_day)).strftime('%Y-%m-%d')

    c = twint.Config()
    #Keyword search for our def.
    c.Search = search_term
    #This is the number of Tweets we want to get in our search.
    c.Limit = limit
    #We want the search data to be saved into a CSV file.
    c.Store_csv = True
    c.Since = since
    #Name of the file we save our data to.
    c.Output = 'data/output.csv'
    c.Pandas = True
    c.Pandas_clean = True
    #This is limiting the search to only English. 
    c.Lang = "en"
    twint.run.Search(c)
    df = twint.storage.panda.Tweets_df 
    return df

# setting up global variables
def column_to_string(df, column_name):
    return df[column_name].astype(str).str.cat(sep=' ')

# Cleaning the tweets Step 2

def write_tweets_to_text_file(text_data):
    with open('text_data.txt', 'wb') as f:
        f.write(text_data, codecs.getwriter('utf-8')(f), ensure_ascii=False)


# Tokenization (2pt)
#We tokenise the words because we can not give a sentiment analysis if we do not have each word separated. 

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
    lemmatized_sentence = " ".join([token.lemma_ for token in doc if len(token) >  3 ])
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
    text = re.sub('’', ' ', text)  # remove apostrophe'
    text = re.sub('\'', ' ', text)  # remove double spacing
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



# This function  converts a column of words into a list of words
def column_to_string(df, column_name):
    return df[column_name].astype(str).str.cat(sep=' ')




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
    # Removing Duplicates
    tweets = tweets.drop_duplicates(subset='tweet', keep="last")

    # Getting the sentiment score of the tweets before preprocessing
    before_preprocessing_sentiment = sentiment_analysis(tweets)
    # print('Sentiment: ' + before_preprocessing_sentiment['Sentiment'].astype(str) + "  Polorization: " + before_preprocessing_sentiment['Polarity'].astype(str))  # Printing the sentiment score of the tweets before preprocessing
    print('Cleaning Tweets...')
    tweets = apply_preprocessing(tweets)
    # print(tweets['cleaned_tweets'])
    # # Sperator
    # print('\n')
    # print('\n')
    # Removing short words
    tweets['cleaned_tweets']= tweets['cleaned_tweets'].apply(lambda x: " ".join ([w for w in x.split() if len (w)>3]))
    # Getting the sentiment score of the tweets after preprocessing
    
    after_preprocessing_sentiment = sentiment_analysis(tweets)
    # print('Sentiment: ' + after_preprocessing_sentiment['Sentiment'].astype(str) + " Polorization: " + after_preprocessing_sentiment['Polarity'].astype(str))  # Printing the sentiment score of the tweets before preprocessing

    df = tweets
    
    # Dumping the cleaned tweets to a CSV file
    print('Writing to CSV file...')
    if (get_sentiment_before_preprocessing):

        before_preprocessing_sentiment.to_csv('./data/Complete.csv')
        before_preprocessing_sentiment.to_json('./data/Complete.json')
        return after_preprocessing_sentiment
    else:
        after_preprocessing_sentiment.to_csv('./data/Complete.csv')
        after_preprocessing_sentiment.to_json('./data/Complete.json')

        return before_preprocessing_sentiment


# Running the main function
final_ouput = main()

# Showing the Polarity of the tweets using a search Term    

  

# Part 4 visualizing the data

# visualizing the data Plot Sentiment of the tweets using a search Term 
def plot_sentiment(table):
    pal = {"positive":'r', "negative":"g","neutral":"b"}
    fig1 = sns.displot(table, x="Sentiment", hue="Sentiment", legend=False, palette= pal)
    fig1.fig.suptitle("Count of tweets by Sentiment",fontsize =15)
    plt.tight_layout()
    plt.show()

# This Function is used to plot the frequency of the words in the tweets
def visualize_term_freq(table):
    data_list = table.loc[:,"cleaned_tweets"].to_list()
    flat_data_list = [sublist.split(' ') for sublist in data_list  ]
    print(flat_data_list)
    data_count= pd.DataFrame(flat_data_list)
    data_count= data_count[0].value_counts()
    freq_count = FreqDist()
    for words in data_count:
        freq_count[words] +=1
        print(words , ' count is ' , freq_count[words])

    # Ploting 
    data_count = data_count[:20,]
    plt.figure(figsize=(10,5))
    sns.barplot(data_count.values, data_count.index, alpha=0.8)
    plt.title('Top Words Overall')
    plt.ylabel('Word from Tweet', fontsize=12)
    plt.xlabel('Count of Words', fontsize=12)
    plt.show()


# Plot Multiple relations of the tweets 
def plot_tables(table):
    # Drop the columns that are not needed
    table = table.drop(['id','timezone', 'place','language', 'hashtags',
        'cashtags', 'user_id', 'username', 'name', 'day', 'hour', 'nlikes',
        'search','conversation_id', 'created_at', 'user_id_str', 'link', 'urls', 'photos', 'video',
        'thumbnail', 'retweet','nreplies', 'nretweets', 'quote_url', 'near', 'geo', 'source', 'user_rt_id', 'user_rt',
        'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src',
        'trans_dest'],axis = 1)
        # Show the remaining table plots 
    sns.pairplot(table, hue='Sentiment', size=2.5);
    plt.show()
    

