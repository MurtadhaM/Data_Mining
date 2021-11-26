#!/usr/local/bin/python3.9

# Author: Murtadha Marzouq
# Date:   12/10/2021
# Time:   12:00 PM
# Assignment: Word Frequency Analysis

import json
import numpy as np
import pandas as pd
import gensim
from gensim.corpora import Dictionary
import spacy
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.probability import FreqDist

tweets = pd.read_csv('data/Completed100.csv')
d = pd.DataFrame()

data_list = tweets.loc[:,"cleaned_tweets"].to_list()
flat_data_list = [sublist.split(' ') for sublist in data_list  ]
print(flat_data_list)
data_count= pd.DataFrame(flat_data_list)
data_count= data_count[0].value_counts()
freq_count = FreqDist()
for words in data_count:
  freq_count[words] +=1
  print(words , ' count is ' , freq_count[words])


data_count = data_count[:20,]
plt.figure(figsize=(10,5))
sns.barplot(data_count.values, data_count.index, alpha=0.8)
plt.title('Top Words Overall')
plt.ylabel('Word from Tweet', fontsize=12)
plt.xlabel('Count of Words', fontsize=12)
plt.show()