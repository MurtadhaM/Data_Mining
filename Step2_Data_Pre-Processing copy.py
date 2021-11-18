#USING THIS GUIDE https://medium.com/nerd-for-tech/step-by-step-guide-to-twitter-sentiment-analysis-bc250caf3a3c
# Pre-processing the data

#!pip3 install twint nest_asyncio
#!pip3 install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
import twint
import nest_asyncio
from textblob import TextBlob
import json, codecs , re
nest_asyncio.apply()



# Instantiate and configure the twint-object
try:  # Run search
    
  c = twint.Config()
  c.Store_object = True
  c.Pandas =True
  c.Search = "#okboomer"
  c.Hide_output=True
  c.Limit = 10
  c.Lang = 'en'
  c.Store_csv = True
  c.Output = "data/Test_Search.csv"
  c.Pandas = True

  twint.run.Search(c)
  df = twint.storage.panda.Tweets_df #result is saved to df  

  # Quick check


  
  print('Columns are')
  print(df.keys())
  print('number of entries:' + str(len(df.values)))
  tweet_text = df['tweet'].to_list()
  print(tweet_text)
  #cleaning the text

  with open('text_data.json', 'wb') as f:
   json.dump(tweet_text, codecs.getwriter('utf-8')(f), ensure_ascii=False)
except Exception as e:
  print(e)
#print(tweet_text)
#extract year,month,day into new columns from datetime column

print(df)
clean_tweets = df['tweet'].to_list()
print(clean_tweets)
"""# New Section"""

df = twint.storage.panda.Tweets_df
print('Columns are')
print(df.keys())
print('number of entries:' + str(len(df.values)))




