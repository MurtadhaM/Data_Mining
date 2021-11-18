#!pip3 install twint nest_asyncio
#!pip3 install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
import twint
import nest_asyncio
import json, codecs
nest_asyncio.apply()



# Instantiate and configure the twint-object
try:
  c = twint.Config()
  c.Store_object = True
  c.Pandas =True
  c.Search = "#okboomer"
  c.Hide_output=True
  c.Pandas_clean=True
  c.Limit = 10
  c.Lang = 'en'
  c.Store_csv = True
  c.Output = "data/Test_Search.csv"

  # Run search
  twint.run.Search(c)

  # Quick check
  twint.storage.panda.Tweets_df.head()


  df = twint.storage.panda.Tweets_df
  print('Columns are')
  print(df.keys())
  print('number of entries:' + str(len(df.values)))
  tweet_text = df['tweet'].to_list()
  print(tweet_text)

  with open('text_data.json', 'wb') as f:
   json.dump(tweet_text, codecs.getwriter('utf-8')(f), ensure_ascii=False)
except Exception as e:
  print(e)
print(tweet_text)

"""# New Section"""

df = twint.storage.panda.Tweets_df
print('Columns are')
print(df.keys())
print('number of entries:' + str(len(df.values)))

