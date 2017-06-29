# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from tweet_regex import tokenize 
import pickle
import h5py




FILE_NAME_TWEETS_BY_KEYWORD = 'tweets_full_complex_new'
FILE_NAME_TWEETS_BY_PAGES   = 'tweets_bypage'
FILE_NAME_COMBINED = 'tweets_combined_ready_for_glove'
 
FILE_NAME_DATE = 'dataset_prepared_very_date' 



'''
###############################################################################
##################          LOAD DATA                     #####################
###############################################################################
'''
#Load tweets scraped by keywords
with open(FILE_NAME_TWEETS_BY_KEYWORD + '.pickle', 'rb') as f:
    df_tweets_by_keywords = pickle.load(f)
    
#Load tweets scraped by pages
with open(FILE_NAME_TWEETS_BY_PAGES + '.pickle', 'rb') as f:
    df_tweets_by_pages = pickle.load(f)

#Load whole date column of preprocessed dataset
with open(FILE_NAME_DATE + '.pickle', 'rb') as f:
    se_dates = pickle.load(f)
  
  
'''
###############################################################################
##################         CONCAT TWEET DATASETS          #####################
###############################################################################
'''
#Some hacking to create consistency
df_tweets_by_pages.rename(columns={'TWEET':'TWEETS'}, inplace=True)    
df_tweets_by_pages['FAVORITES'] = 10000
df_tweets_by_pages['RETWEETS'] = 10000

#Concatenating
df_tweets = pd.concat([df_tweets_by_keywords, df_tweets_by_pages])
 
#Date format  
df_tweets['DATE'] = pd.to_datetime(df_tweets['DATE'])  

print('Start date of dataset: ', se_dates.min())
print('End date of dataset: ', se_dates.max())
print('Start date of tweets: ', df_tweets['DATE'].min())
print('End date of tweets: ', df_tweets['DATE'].max())


'''
###############################################################################
##################          FILTERING TWEETS              #####################
###############################################################################
'''
#Select relevant time interval
date_min = se_dates.min()
date_max = se_dates.max()
df_tweets = df_tweets[(df_tweets['DATE'] >= date_min) & (df_tweets['DATE'] <= date_max)] #between given date


#Sort tweets by relevance
df_tweets = df_tweets.groupby('DATE').apply(lambda x: x.sort_values(by=['FAVORITES','RETWEETS'], ascending=False))


#Select only top 100 per day
df_tweets_padded = df_tweets.groupby('DATE').apply(lambda x: x[:100])                                    # first 100
print('Start date of dataset: ', se_dates.min())
print('End date of dataset: ', se_dates.max())
print('Start date of tweets: ', df_tweets['DATE'].min())
print('End date of tweets: ', df_tweets['DATE'].max())



'''
###############################################################################
##################          REGEX TWEETS                  #####################
###############################################################################
'''
#Regex tweet texts
df_tweets_padded['TWEETS'] = df_tweets_padded['TWEETS'].map(tokenize)


'''
###############################################################################
##################          STATISTICS                    #####################
###############################################################################
'''
#Some statistics about word number
se_nb_words = df_tweets_padded['TWEETS'].map(lambda x: len(x.split()))
avg_word_nb = np.ceil(se_nb_words.mean())
print('Avg of word numbers (ceiled) is: {}'.format(avg_word_nb))


'''
###############################################################################
##################          SAVE TO DISK                  #####################
###############################################################################
'''
new_file_name = FILE_NAME_COMBINED + '.pickle'
with open(new_file_name, 'wb') as f:
    pickle.dump(df_tweets_padded, f, -1)

print('Preprocessing has happened')
print('Preprocessed tweets are saved to disk, with name: ', new_file_name)
