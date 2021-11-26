#!/usr/local/bin/python3.9

# Author: Murtadha Marzouq
# Date:   12/10/2021
# Time:   12:00 PM
# Assignment: Word Frequency Analysis
import doctest
import sys
import re
from collections import Counter
import pandas as pd


# This function  converts a column of words into a list of words
def column_to_string(df, column_name):
    return df[column_name].astype(str).str.cat(sep=' ')



# this function counts the number of words in a string
def get_word_frequency(text):
    count = Counter()
    # clean the text
    cleaner = re.compile("[a-zA-Z_][a-zA-Z0-9-_]+")
    for line in text:
        count.update(re.findall(cleaner, line))
    wordCount = Counter()
    for ident in count:
        value = count[ident]
        words = [x.lower()for x in re.findall("[A-Z]*[a-z]+(?=[A-Z-_]|$)", ident)]
        wordCount.update({w: value for w in words})
    for ident, value in wordCount.most_common(10):
        print( (ident, value))
    return wordCount
        
        
        
tweets = pd.read_csv('data/Completed100.csv')
text  = column_to_string(tweets, 'cleaned_tweets').split(' ')

word_frequency = get_word_frequency(text)
print(word_frequency)
