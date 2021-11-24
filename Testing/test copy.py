#!//usr/local/bin/python3.9
from nltk import tokenize
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfTransformer, TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline
import stopwords 
from glob import glob
import re
import string
import funcy as fp
from gensim import models
from gensim.corpora import Dictionary, MmCorpus
import nltk
import pandas as pd

from nltk.corpus import stopwords
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS, TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
import pickle



tok = nltk.toktok.ToktokTokenizer()
df = pd.read_csv('data/Complete500.csv')#getting rid of null values
corpus = df['cleaned_tweets']
each_word = []

words = " "
tokens = tok.tokenize(corpus)
for word in corpus:
        if word is not None and  word != " " and len(word) > 3:
            each_word.append(word)
            print(word)

vocabulary = tok.tokenize(each_word)


# new_df = corpus.apply(lambda x: x.split(' '))
print(each_word)

#print()




#print(tfidf_vectorizer)

# pipeline = Pipeline(steps= [('tfidf', TfidfVectorizer(lowercase=True,
#                                                       max_features=1000,
#                                                       stop_words= ENGLISH_STOP_WORDS)),
#                                                       ('model', RandomForestClassifier(n_estimators = 100))])

# fit the pipeline model with the training data                            
#pipeline.fit(df.cleaned_tweets, df.Sentiment)

from nltk.probability import FreqDist
fdist = FreqDist(each_word)
print(fdist.most_common(10))
print(len(words))
print(len(vocabulary))
