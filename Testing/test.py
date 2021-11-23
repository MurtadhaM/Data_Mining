#!//usr/local/bin/python3.9
from nltk import tokenize
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline

tok = nltk.toktok.ToktokTokenizer()
df = pd.read_csv('data/Complete500.csv')#getting rid of null values
corpus = df['cleaned_tweets'].values

words = " "
tokens = tok.tokenize(corpus)
for word in tokens:
    if word is not None and  word != " " and len(word) > 3:
        words = words + " " + word
vocabulary = tok.tokenize(words)
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(vocabulary)
print(vocabulary)
vectorizer.get_feature_names_out()

print(X.toarray())
# print(words)
# print(tokens)
# print(len(tokens))
vectorizer2 = CountVectorizer(analyzer='word', ngram_range=(1, 2))
X2 = vectorizer2.fit_transform(vocabulary)
print(vectorizer2.get_feature_names_out())
print(X2.toarray())
# pipe = Pipeline([('count', CountVectorizer(vocabulary=vocabulary)),('tfid', TfidfTransformer())]).fit(vocabulary) 

# #df = df.dropna()#Taking a 30% representative sample
# import numpy as np
# df1 = df
