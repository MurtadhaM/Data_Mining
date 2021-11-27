#!/usr/local/bin/python3.9

# Author: Murtadha Marzouq
# Date:   12/10/2021
# Time:   12:00 PM
# Assignment: Social Media Analysis
# pip3 install --user --upgrade git+https://github.com/twintproject/twint.git@origin/master#egg=twint
import  pandas as pd
import matplotlib as plt
import seaborn as sns

dataframe = pd.read_csv('./data/Complete.csv')

# Part 4 visualizing the data
# visualizing the data Plot Sentiment of the tweets using a search Term 
def plot_sentiment(table):
    pal = {"positive":'r', "negative":"g","neutral":"b"}
    fig1 = sns.displot(table, x="Sentiment", hue="Sentiment", legend=False, palette= pal)
    fig1.fig.suptitle("Count of tweets by Sentiment",fontsize =15)
    plt.tight_layout()
    plt.show()

# This Function is used to plot the frequency of the words in the tweets
def visualize_term_freq(table):
    data_list = table.loc[:,"cleaned_tweets"].to_list()
    flat_data_list = [sublist.split(' ') for sublist in data_list  ]
    print(flat_data_list)
    data_count= pd.DataFrame(flat_data_list)
    data_count= data_count[0].value_counts()
    freq_count = FreqDist()
    for words in data_count:
        freq_count[words] +=1
        print(words , ' count is ' , freq_count[words])

    # Ploting 
    data_count = data_count[:20,]
    plt.figure(figsize=(10,5))
    sns.barplot(data_count.values, data_count.index, alpha=0.8)
    plt.title('Top Words Overall')
    plt.ylabel('Word from Tweet', fontsize=12)
    plt.xlabel('Count of Words', fontsize=12)
    plt.show()


# Plot Multiple relations of the tweets 
def plot_tables(table):
    # Drop the columns that are not needed
    table = table.drop(['id','timezone', 'place','language', 'hashtags',
        'cashtags', 'user_id', 'username', 'name', 'day', 'hour', 'nlikes',
        'search','conversation_id', 'created_at', 'user_id_str', 'link', 'urls', 'photos', 'video',
        'thumbnail', 'retweet','nreplies', 'nretweets', 'quote_url', 'near', 'geo', 'source', 'user_rt_id', 'user_rt',
        'retweet_id', 'reply_to', 'retweet_date', 'translate', 'trans_src',
        'trans_dest'],axis = 1)
        # Show the remaining table plots 
    sns.pairplot(table, hue='Sentiment', size=2.5);
    plt.show()
    

