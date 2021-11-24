import pandas as pd
import sklearn as sk
import math 
import nltknltk.download('stopwords')
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()

response = tfidf.fit_transform([docA, docB])