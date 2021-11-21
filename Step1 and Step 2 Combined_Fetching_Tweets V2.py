# Author: Murtadha Marzouq
# Date:   12/10/2021
# Time:   12:00 PM
# Assignment: Social Media Analysis
from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import re
import os
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


# Setting up the Tweepy API to pull tweets
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
print(df)
df.head()
df['language'].value_counts()
df = df[df['language'] == 'en']

tweets = df[['tweet']]





# Cleaning the tweets Step 2 



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


# lemmatization
def lemmatize_text(input):
    tagged_tokens = nltk.pos_tag(
        nltk.word_tokenize(input))  # Positonal tagging
    wordnet_tokens = pos_tag_wordnet(tagged_tokens)
    lemmatized_text = [wnl.lemmatize(word, pos)
                       for word, pos in wordnet_tokens]
    return lemmatized_text


def clean_tweets_tb(input):
    text = str(input)
    text = re.sub(r'https?://\S+', '', text)
    text = re.sub("@[A-Za-z0-9]+","",text) 
    text = re.sub(r"@[A-Za-z0-9]+", "", text)  
    text = re.sub(r"(?:\@|http?\://|https?\://|www)\S+", "", text)
    text = re.sub(r"_[A-Za-z0-9]+", "", text)  
    text = re.sub(r"__", "", text)  
    text = re.sub(' +', ' ', text)  
    text = "".join([char for char in text if char not in string.punctuation])
    text = text.lower()  # Lower text
    return text

# Applying PreProcessing

def apply_preprocessing():
    tweets['tweet_tb'] = tweets['tweet'].apply(tokenize)
    tweets['tweet_tb'] = tweets['tweet_tb'].apply(clean_tweets_tb)
    tweets['tweet_tb'] = tweets['tweet_tb'].apply(lambda x: lemmatize_text(x))
    stop = nltk.corpus.stopwords.words('english')
    stop.extend(["amp", "https", "co", "rt", "new", "let",
                "also", "still", "one", "people", "gt"])
    tweets['tweet_tb'] = tweets['tweet_tb'].apply(
        lambda x: " ".join(x for x in str(x).split() if x not in stop))
    return tweets

# Applying PreProcessing
output = apply_preprocessing()

# Printing
output['tweet_tb'].apply(print)
