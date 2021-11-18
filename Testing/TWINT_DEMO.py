import twint
nest_asyncio.apply()


# Instantiate and configure the twint-object
c = twint.Config()
c.Store_object = True
c.Pandas =True
c.Search = "#okboomer"
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
df

#DUMPING THE TEXT INTO A JSON FILE TO PROCESSING:
tweet_text = df['tweet'].to_list()


print(tweet_text)