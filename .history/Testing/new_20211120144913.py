from __future__ import print_function
import pyLDAvis
import pyLDAvis.sklearn
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from nltk.tokenize import word_tokenize
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()
with open('text_data.json', 'r') as f:
    text = f.read()




# Word stop the text
bad_words = stopwords.words('english')
print(bad_words)
words = [word for word in nltk.word_tokenize(text, language='english') if word.lower() not in bad_words]
new_text = " ".join(words)

# Tokenize the text
tokenizer = nltk.word_tokenize(new_text, language='english')

# Lemmatize the text
for token in new_text:
    lemmatizer.lemmatize(new_text) 
    print(lemmatizer.lemmatize(token))

print(lemmatizer.lemmatize(tokenizer[10]))
print(tokenizer)

print("Old length: ", len(text))
print("New length: ", len(new_text))