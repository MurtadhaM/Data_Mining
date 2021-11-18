import nest_asyncio
import twint

nest_asyncio.apply()


# Instantiate and configure the twint-object
try:
  c = twint.Config()
  c.Store_object = True
  c.Pandas =True
  c.Search = "#okboomer"
  c.Limit = 10
  c.Lang = 'en'
  c.Store_csv = True
#  c.Output = "data/Test_Search.csv"

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
except Exception as e:
  print(e)

print(tweet_text)

"""# New Section"""

df = twint.storage.panda.Tweets_df
print('Columns are')
print(df.keys())
print('number of entries:' + str(len(df.values)))
df

#DUMPING THE TEXT INTO A JSON FILE TO PROCESSING:
tweet_text = df['tweet'].to_list()

df['date_only'] = pd.to_datetime(df['date'])
date_only = df['date_only'].dt.date
num_days = len(date_only.unique())
print("Number of days jack has tweeted:", num_days)

tweets = df['tweet'].to_list()

words = ''
stopwords = set(STOPWORDS)

for value in tweets:
    value = str(value)
    tokens = value.split()
    for i in range(len(tokens)):
        tokens[i] = tokens[i].lower()
        
    words += " ".join(tokens)+" "

wordcloud = WordCloud(width = 1000, height = 1000,
                     background_color = 'grey',
                     stopwords = stopwords,
                     min_font_size = 10).generate(words)

plt.figure(figsize=(8,8))
plt.imshow(wordcloud)
plt.axis("off")
plt.show()
