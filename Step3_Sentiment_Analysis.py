import pandas as pd
import numpy as np
import seaborn as sns
from textblob import TextBlob , classifiers
import matplotlib.pyplot as plt
# Setting up a TextBlob object
textblob = pd.read_csv('data/Test_Search.csv')
textblob = pd.DataFrame(textblob)


# #### Step 3: Sentiment Analysis (8pts total) ####

def sentiment_analysis(text):
    blob = TextBlob(text)
    print(blob.polarity)
    print("The test case is '" + text + "' and the sentiment calculation is" + str(blob.sentiment))

    return blob.sentiment 


for tweet in textblob.tweet:
    sentiment_analysis(tweet)
final_df = pd.DataFrame(columns=['tweet', 'Sentiment', 'Polarity'])
final_df['Polarity'] = (textblob['tweet'].map(lambda tweet: TextBlob(tweet).sentiment.polarity))
final_df["Sentiment"] = final_df["Polarity"].map(lambda pol: '+' if pol > 0 else '-')
positive = final_df[final_df.Sentiment == "+"].count()["tweet"]
negative = final_df[final_df.Sentiment == "-"].count()["tweet"]
print(final_df)


































