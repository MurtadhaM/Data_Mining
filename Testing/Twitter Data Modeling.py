#!//usr/local/bin/python3.9
from nltk import tokenize
import pandas as pd
import nltk
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.pipeline import Pipeline



df = pd.read_csv('data/Test.csv')#getting rid of null values
corpus = df['Tweets']
print(corpus) 




