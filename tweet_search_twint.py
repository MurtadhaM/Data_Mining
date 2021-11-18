import nest_asyncio
import twint

nest_asyncio.apply()


# Instantiate and configure the twint-object
c = twint.Config()
c.Store_object = True
c.Pandas =True
c.Search = "#okboomer"
c.Limit = 10000
c.Lang = 'en'
c.Store_csv = True
c.Output = "search_okboomer.csv"

# Run search
twint.run.Search(c)

# Quick check
twint.storage.panda.Tweets_df.head()

"""### The End (of the data extraction)
the stuff below is just some cleanup...
"""

# Cleanup
tweets = twint.storage.panda.Tweets_df.drop_duplicates(subset=['id'])
print(len(tweets))

# Reindex
tweets.index = range(len(tweets))

# Remove non-english
from whatthelang import WhatTheLang
wtl = WhatTheLang()

# This function makes easy to handle exceptions (e.g. no text where text should be)
# not really needed but can be useful 

def detect_lang(text):
    try: 
        return wtl.predict_lang(text)
    except Exception:
        return 'exp'

# Commented out IPython magic to ensure Python compatibility.
# # Added performance measure here...you can leave teh %%time line out
# 
# %%time
# 
# tweets['lang'] = tweets['tweet'].map(lambda t: detect_lang(t))

# keep only english

tweets = tweets[tweets.lang == 'en']
print(len(tweets))

tweets.head()

# Connect Google drive
from google.colab import drive
drive.mount('/content/drive')

# Done

tweets.to_csv("drive/My Drive/Colab Notebooks/tweets_boomer.csv")

"""## Get peoples' connections

This is a short analysis in which I combine (very) basic scraping with extraction of Twitter networks and network analysis. 
The purpose was to identify interesting people on Twitter for me to follow...

The appropach:

- Fet links to all shows
- Fetch links to twitter-accounts form the shownotes
- Use these URLs to identify users
- Scrape all people these people follow

Assumption: People that are followed by people that are invited on TwimlAI are people, I should be following...
"""

# Import libraries
import re
import pickle # pickle is for storing element...pickling... you can store any kind of python object with that
import requests as rq

# Load HTML parser library...yes, that's its name.
from bs4 import BeautifulSoup

# Get URLs of all TWIML shows
r = rq.get('https://twimlai.com/shows/')

# Parse the HTML
soup = BeautifulSoup(r.text)

# Fetch all links from parsed HTML
links = soup.find_all('a')

# Keep only links leading to a twiml-podcast
links = [l.attrs['href'] for l in links if l.attrs['href'].startswith('https://twimlai.com/twiml-talk')]

# Drop duplicated links
links = list(set(links))

# Iterate and fetch show-notes, then extract links leading to twitter. 
twitter_urls = []
for link in links:
  show = rq.get(link) # get shownotes 
  soup = BeautifulSoup(show.text) # parse
  show_links = soup.find_all('a') # find links 
  show_links = [l.attrs['href'] for l in show_links if l.attrs['href'].startswith('https://twitter.com')] # keep only links to twitter
  twitter_urls.extend(show_links) # store

# Store the lovely list of links to twitter profiles
pickle.dump(list(set(twitter_urls)), open('twitter-list.p','wb'))

# Unless already imported
import twint
import numpy as np

# Filter out tooooo long twitter links that are more than likely not profiles
usernames = [x.replace('https://twitter.com/','') for x in set(twitter_urls) if len(x) <= 50]

# Profile lookup

for username in usernames:
  c = twint.Config()
  c.Username = username
  c.Store_object = True
  c.User_full = False
  c.Pandas =True
  twint.run.Lookup(c)

# Store in a DF
user_df = twint.storage.panda.User_df.drop_duplicates(subset=['id'])

#Store away
user_df.to_csv('user_df.csv')

# Or like that
user_df[['bio','username']].to_csv('short.csv')

# Clean up
twint.storage.panda.clean()
twint.output.clean_follow_list()

# Connect Google drive
from google.colab import drive
drive.mount('/content/drive')

"""Unfortunately getting followers is not as easy. It requires some trickery. In this case I decided to write out the followers of each person as a pickle file to disk. This happened after I realized that I often get blank responses. Writing on disk of individual DFs with followers allowed me to spot the ones that are empty and remove by hand. Probably there is a smarter solution to that somewhere"""

# we iterate over the different usernames and store follower dataframe
for u in user_df['username']:
  c = twint.Config()
  c.Username = u
  c.Store_object = True
  c.User_full = False
  c.Pandas = True
  c.Store_pandas = True
  c.Stats = False
  c.Hide_output = True

  twint.run.Following(c)
  twint.storage.panda.Follow_df.to_pickle("/content/drive/My Drive/Colab/TWIML-guests/{}.p".format(u))
  twint.storage.panda.clean()
  twint.output.clean_follow_list()

# To get the data back we use glob...which will help us dealing with many tiny filed
import glob

# Get paths of all stored files
paths = glob.glob('/content/drive/My Drive/Colab/TWIML-guests/*.*')

# Create an edgelist
# read stored DFs with following, append into long edgelist

empty = []
edgelist = pd.DataFrame(columns = ['target', 'source'])
for path in paths:
  df = pd.read_pickle(path)
  if len(df) == 1:
    name = df.index[0]
    edges = pd.DataFrame(df['following'][name], columns=['target'])
    edges['source'] = name
    edgelist = edgelist.append(edges)
  else:
    empty.append(path)

# Reindex

edgelist.index = range(edgelist.shape[0])

"""### From here: Network analysis 101"""

import networkx as nx
from networkx.algorithms import bipartite

G = nx.DiGraph()

G.add_edges_from([(u,v) for (u,v) in zip(edgelist['source'],edgelist['target'])])

len(G.nodes)

eigenvector = nx.eigenvector_centrality(G)

nx.set_node_attributes(G, eigenvector, 'eigenvector_centrality')

import community

G_und = G.to_undirected()

communities = community.best_partition(G_und, resolution = 1)
nx.set_node_attributes(G, communities, 'community')

perc_filter = np.percentile([v for u,v in eigenvector.items()], 90)

nodes_selected = [x for x,y in eigenvector.items() if y >= perc_filter]

G_sub = G.subgraph(nodes_selected)

communities = community.best_partition(G_sub.to_undirected(), resolution = 1)
nx.set_node_attributes(G_sub, communities, 'community_2')

len(G_sub.nodes)

nx.write_gexf(G_sub, 'twiml.gexf')

net_df = pd.DataFrame(dict(G_sub.nodes(data=True))).T

net_df.groupby('community_2').apply(lambda t: t.sort_values(['eigenvector_centrality'],ascending=False)[:10])

nlp_ppl = net_df[net_df.community_2 == 3].sort_values(['eigenvector_centrality'],ascending=False).index