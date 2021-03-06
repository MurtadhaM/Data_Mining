# Author: Murtadha Marzouq
# Date:   12/10/2021
# Time:   12:00 PM
# Assignment: Social Media Analysis

import spacy
import nltk 
import codecs
from nltk.stem import WordNetLemmatizer
nltk.download('wordnet')
from nltk.corpus import wordnet
wnl = WordNetLemmatizer()
nlp = spacy.load('en_core_web_sm')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
import re
import pandas as pd
import pyLDAvis
from imp import reload
import pyLDAvis.sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import json
# This Needs The Tweet Data Exported From Part 1 File to a text_data.json file in the same directory
# Pre-processing the tweets/articles

### Step 2: Pre-processing the tweets/articles (8pts total) ####


# read the data from a file
df = pd.read_csv('data/Complete500.csv')#getting rid of null values
text  = df['cleaned_tweets'].values
        
print(text)

#Tokenization (2pt)
def tokenize(text):
    tok = nltk.toktok.ToktokTokenizer()
    tokens = tok.tokenize(text)
    print('No of tokens:', len(tokens) )
    return tokens
# Lemmatization (2pt)
## Positional Tagging the word
def pos_tag_wordnet(tagged_tokens):
    tag_map = {'j': wordnet.ADJ, 'v': wordnet.VERB, 'n': wordnet.NOUN, 'r': wordnet.ADV}
    new_tagged_tokens = [(word, tag_map.get(tag[0].lower(), wordnet.NOUN))
                            for word, tag in tagged_tokens]
    return new_tagged_tokens

#lemmatization
def lemmatize_text(input):
    tagged_tokens = nltk.pos_tag(nltk.word_tokenize(input)) #Positonal tagging
    wordnet_tokens = pos_tag_wordnet(tagged_tokens)
    lemmatized_text = [wnl.lemmatize(word, pos) for word, pos in wordnet_tokens]
    return lemmatized_text



# removing standard stopwords (2pts)

#  removing additional stopwords as you see fit (2pts)
# Stopwords removal
def stopwords(input):
    print('No of tokens before stopwords removal:', len(input))
    stop_words = set(nltk.corpus.stopwords.words('english'))
    filtered_sentence = [w for w in input if not w in stop_words]
    print('No of tokens after stopwords removal:', len(filtered_sentence))
    return filtered_sentence
 # Running the above functions


# DATA CLEANING
def clean_data(text):
    # Lowercase words
    clean_data = [str(word).lower() for word in text]
    # Eliminate StopWords
    stop_words = set(nltk.corpus.stopwords.words('english'))
    clean_data  = " ".join([word for word in str(text).split() if word not in stop_words])
    # Eliminates Emails
    clean_data = [re.sub(r'\S*@\S*\s?', '', str(word)) for word in text]
    # Eliminate new line
    clean_data = [re.sub(r'\s+', ' ', str(word)) for word in text]
    # Eliminate single quotes
    clean_data = [re.sub("\'", "", word) for word in text]
    # Eliminates words where len(word) < 3
    clean_data = [' '.join([word for word in word.split() if len(word)>4]) for word in text]
    return clean_data

def data_cleaning(df_tweets):
    '''Clean the Tweets'''
    # convert to lower case
    df_tweets['clean_text'] = df_tweets['tweets'].str.lower()
    # Remove punctuations
    df_tweets['clean_text'] = df_tweets['clean_text'].str.replace('[^\w\s]',' ')
    # Remove spaces in between words
    df_tweets['clean_text'] = df_tweets['clean_text'].str.replace(' +', ' ')
    # Remove Numbers
    df_tweets['clean_text'] = df_tweets['clean_text'].str.replace('\d+', '')
    # Remove trailing spaces
    df_tweets['clean_text'] = df_tweets['clean_text'].str.strip()

    df_tweets['clean_text'] = df_tweets['clean_text'].str.strip()
    # Remove URLS
    # remove stop words
    stop = stopwords('english')
    stop.extend(["amp","https","co","rt","new","let","also","still","one","people","gt"])
    df_tweets['clean_text'] =  df_tweets['clean_text'].apply(lambda x: " ".join(x for x in x.split() if x not in stop ))
    return df_tweets


df['tweets'] = ''
df['clean_text'] = ''

df['tweets'] = pd.read_csv('data/Complete500.csv')['tweet']
#print(df)
#print(df)
# tokens = tokenize(text)
#filtered_sentence = stopwords(tokens)
#filtered_sentence_text = ' '.join(filtered_sentence)
#lemma_words = lemmatize_text(text)
#test = clean_data(lemma_words)
data = data_cleaning(df)
print(data)
# 

words = []
for word in data['clean_text'] :
    for w in word.split(' '):
        words.append(w)
print(words)
#for word in df['clean_text'] :
 #   print(word)


