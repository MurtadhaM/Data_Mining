#!/usr/local/bin/python3.9

import re
from gensim import corpora
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
import pandas as pd
import matplotlib.pyplot as plt
stopwords = nltk.corpus.stopwords.words('english')
stop_words = stopwords

stemmer = LancasterStemmer()    


df = pd.read_csv('data/Completed100.csv')


def remove_content(text):
    text = re.sub(r"http\S+", "", text) #remove urls
    text=re.sub(r'\S+\.com\S+','',text) #remove urls
    text=re.sub(r'\@\w+','',text) #remove mentions
    text =re.sub(r'\#\w+','',text) #remove hashtags
    return text
def process_text(text, stem=False): #clean text
    text=remove_content(text)
    text = re.sub('[^A-Za-z]', ' ', text.lower()) #remove non-alphabets
    tokenized_text = word_tokenize(text) #tokenize
    clean_text = [
         word for word in tokenized_text
         if word not in stop_words
    ]
    if stem:
        clean_text=[stemmer.stem(word) for word in clean_text]
    return ' '.join(clean_text)


r = [process_text(x,stem=False).split() for x in df['tweet'].tolist()] 
dictionary = corpora.Dictionary(r)
corpus = [dictionary.doc2bow(rev) for rev in r]#initialize model and print topics
from gensim import models
model = models.ldamodel.LdaModel(corpus, num_topics=10, id2word=dictionary, passes=15)
topics = model.print_topics(num_words=5)
for topic in topics:
    print(topics[0],process_text(topic[1]))
    
labels=[]
for x in model[corpus]:
    labels.append(sorted(x,key=lambda x: x[1],reverse=True)[0][0])
    df['topic']=pd.Series(labels)
    
data = df
data['clean_text'] = df.cleaned_tweets
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer(stop_words=stop_words)
model = vectorizer.fit(data.clean_text)
docs = vectorizer.transform(data.clean_text)
lda = LatentDirichletAllocation(20)
lda.fit(docs)
def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([(feature_names[i])
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()
    
print_top_words(lda,vectorizer.get_feature_names(),10)

data['topic']=lda.transform(docs).argmax(axis=1)
data.topic.value_counts(normalize=True).plot.bar()
print(data['topic'])



data['date']=pd.to_datetime(data.date)
data.index = data.date
for i in range(10):
    temp = data[data.topic==i]
    temp.resample('7D').size().plot()
    plt.title('topic %s' %i)
import pyLDAvis.gensim
from gensim.models import CoherenceModel, LdaModel, LsiModel, HdpModel
ldamodel = LdaModel(corpus=corpus, num_topics=10, id2word=dictionary)
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
ldamodel.show_topics()
print(ldamodel.show_topics())
pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary)
ldatopics = [[word for word, prob in topic] for topicid, topic in ldamodel.show_topics(formatted=False)]
lda_coherence = CoherenceModel(topics=ldatopics, texts=texts, dictionary=dictionary, window_size=10).get_coherence()
print(ldatopics)