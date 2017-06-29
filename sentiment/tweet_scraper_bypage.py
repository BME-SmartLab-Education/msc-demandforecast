# -*- coding: utf-8 -*-


from twitterscraper import query_tweets
from twitterscraper.query import query_all_tweets
import time
import pandas as pd
import sys
import os
from datetime import datetime



#Time interval
file_name = 'tweets_bypages_{}.h5'.format(datetime.now().strftime("%Y_%m_%d-%H_%M_%S")) #like: tweets_2014-10-17_2014-10-18
query = 'from:{}'




def add_tweets_to_collection(ls_tw, tws):
    for tweet in tws:
        ls_data = [
                   tweet.id,
                   tweet.timestamp.strftime('%Y-%m-%d'),
                   tweet.user,
                   tweet.fullname,
                   tweet.text.encode('utf-8')
                   ]
        ls_tw.append(ls_data)




#Download tweets
ls_all_tweets = []  
start = time.time()  
try:
    for page_name in ['ABC1', 'ABC2', 'ABC3']:
        start_actual = time.time()
        tweets = query_all_tweets(query.format(page_name))
        add_tweets_to_collection(ls_all_tweets, tweets)
        duration_actual = time.time()-start_actual
        print('Run took {} minutes for page {}'.format(duration_actual/60, page_name))
except(KeyboardInterrupt):
    pass

except:
    pass
duration = time.time()-start


df_tweets = pd.DataFrame(ls_all_tweets, columns=[
                                                 'ID',
                                                 'DATE',
                                                 'USERNAME',
                                                 'FULLNAME',
                                                 'TWEET'
                                                        ])



#Drop duplicates
df_tweets.drop_duplicates('ID', inplace=True)
print('Dupicates are dropped.')


#Save tweets
df_tweets.to_hdf(file_name, 'tweets')
print('File saved to disk with name: {}. It contains {} tweets.'.format(file_name,df_tweets.shape[0]))
print('Running took {} minutes in all.'.format(duration/60))
    
