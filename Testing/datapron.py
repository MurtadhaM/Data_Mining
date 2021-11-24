import gensim
import gensim.corpora as corpora
from gensim.corpora import Dictionary
from gensim.models.coherencemodel import CoherenceModel
from gensim.models.ldamodel import LdaModel

from pprint import pprint

import spacy

import pickle
import re 
import pyLDAvis

import matplotlib.pyplot as plt 
import pandas as pd


tweets = pd.read_csv('data/Test.csv') #Change this with the name of your downloaded file
tweets = tweets.Tweets.values.tolist()

# Turn the list of string into a list of tokens
tweets = [t.split(',') for t in tweets]

id2word = Dictionary(tweets)
# Term Document Frequency
corpus = [id2word.doc2bow(text) for text in tweets]
print(corpus[:1])
print([[(id2word[i], freq) for i, freq in doc] for doc in corpus[:1]])

lda_model = LdaModel(corpus=corpus,
                   id2word=id2word,
                   num_topics=10, 
                   random_state=0,
                   chunksize=100,
                   alpha='auto',
                   per_word_topics=True)

pprint(lda_model.print_topics())
doc_lda = lda_model[corpus]

lda_model.print_topics()
lda_model.show_topics()

vis = pyLDAvis.prepare(lda_model, corpus, id2word)

vis.show()
