#!/usr/local/bin/python3.9

from gensim.corpora import Dictionary
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
import numpy as np
import gensim
import matplotlib.pyplot as plt
from sklearn.svm import SVC
from gensim import corpora
import pyLDAvis
import pyLDAvis.gensim_models as gensimvis
#pyLDAvis.enable_notebook()
import pandas as pd
import spacy
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
nlp = spacy.load('en_core_web_sm')

stop_words = set(stopwords.words('english'))
data = pd.read_csv('/Users/m/Documents/GitHub/Data_Mining/data/Complete.csv')


def remove_content(text):
    text = re.sub(r"http\S+", "", text)  # remove urls
    text = re.sub(r'\S+\.com\S+', '', text)  # remove urls
    text = re.sub(r'\@\w+', '', text)  # remove mentions
    text = re.sub(r'\#\w+', '', text)  # remove hashtags
    return text


def process_text(text, stem=False):  # clean text
    text = remove_content(text)
    text = re.sub('[^A-Za-z]', ' ', text.lower())  # remove non-alphabets
    tokenized_text = word_tokenize(text)  # tokenize
    clean_text = [
        word for word in tokenized_text
        if word not in stop_words
    ]
    if stem:
        clean_text = [stemmer.stem(word) for word in clean_text]
    return ' '.join(clean_text)


r = [process_text(x, stem=False).split() for x in data['tweet'].tolist()]
dictionary = corpora.Dictionary(r)
# initialize model and print topics
corpus = [dictionary.doc2bow(rev) for rev in r]
doc_term_matrix = [dictionary.doc2bow(rev) for rev in r]
LDA = gensim.models.ldamodel.LdaModel


# Build LDA model
lda_model = LDA(corpus=doc_term_matrix, id2word=dictionary, num_topics=7, random_state=100,
                chunksize=1000, passes=50)
lda_model.print_topics()

print(lda_model.show_topics())
vis = gensimvis.prepare(lda_model, doc_term_matrix, dictionary)

pyLDAvis.save_html(vis, 'LDA_Visualization.html')
print(vis)

