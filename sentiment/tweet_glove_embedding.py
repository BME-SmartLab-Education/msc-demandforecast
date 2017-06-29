# -*- coding: utf-8 -*-


from __future__ import print_function

import os
import sys
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical
from keras.layers import Dense, Input, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model, Sequential
import pandas as pd
from time import time
import h5py
import pickle

START = time()

TWEET_PER_DAY = 100
MAX_SEQUENCE_LENGTH = 20
MAX_NB_WORDS = 20000
EMBEDDING_DIM = 25

TWEET_FILE = 'tweets_combined_ready_for_glove.pickle'

#Glove: twitter
GLOVE_DIR = 'glove_twitter_27B'
GLOVE_FILE = 'glove.twitter.27B.25d.txt'



'''
###############################################################################
##################          GLOVE {WORD:EMBEDDING} MAP             ############
###############################################################################
'''
print('Indexing word vectors..')

embeddings_index = {}
f = open(os.path.join(GLOVE_DIR, GLOVE_FILE))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs
f.close()
print('Found {} word vectors with {} dim.'.format(len(embeddings_index), len(embeddings_index[list(embeddings_index.keys())[0]])))



'''
###############################################################################
##################          TEXTS TO LIST             #########################
###############################################################################
'''
#Load tweets
with open(TWEET_FILE, 'rb') as f:
    df_tweets = pickle.load(f, encoding='latin1') #python2->3
#    df_tweets = pickle.load(f)


texts = df_tweets['TWEETS'].tolist()
DATES_UNIQUE = df_tweets['DATE'].unique()    #Unique dates sorted

print('Found {} tweets.'.format(len(texts)))


'''
###############################################################################
##################          THRESHOLD AND SEQUENCE      #######################
###############################################################################
'''
#Vectorize the text samples into a 2D integer tensor
tokenizer = Tokenizer(nb_words=MAX_NB_WORDS)      # 20.0000
tokenizer.fit_on_texts(texts)                     # list: row  -->  like: [this is apple]
sequences = tokenizer.texts_to_sequences(texts)   # list: row  * word number in that tweet -->  like: [[5,2,32],[32,565,65,1]]

word_index = tokenizer.word_index                 #174.096 
print('Found {} unique tokens.'.format(len(word_index)))

# Padding the sequences to have same length
data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)  #row * 100


'''
###############################################################################
##################          LABELIZING TARGET ATTR      #######################
###############################################################################
'''
#Labelize labels
#labels = to_categorical(np.asarray(labels))
#print('Shape of data tensor:', data.shape)
#print('Shape of label tensor:', labels.shape)



'''
###############################################################################
##################          SPLIT DATASET             #########################
###############################################################################
'''
#Shuffling index
#indices = np.arange(data.shape[0])
#np.random.shuffle(indices)
#data = data[indices]
#labels = labels[indices]


#Split dataset
#num_validation_samples = int(VALIDATION_SPLIT * data.shape[0])
#x_train = data[:-num_validation_samples]
#y_train = labels[:-num_validation_samples]
#x_val = data[-num_validation_samples:]
#y_val = labels[-num_validation_samples:]



'''
###############################################################################
##################          GLOVE MAPPING             #########################
###############################################################################
'''
print('Preparing embedding matrix.')
# prepare embedding matrix
num_words = min(MAX_NB_WORDS, len(word_index) + 1)                  #word_index=original number of diff words
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))             #20.000(!)*25
for word, i in word_index.items():
    if i >=  MAX_NB_WORDS:                                          #20.000 (threshold)
        continue
    embedding_vector = embeddings_index.get(word)                   #1x25 = 1 word vector
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector

print('Embedding matrix shape: ', embedding_matrix.shape)



'''
###############################################################################
##################          RESHAPE TWEETS AND MERGE      #####################
###############################################################################
'''
#Reshape: 433*2000 (433 nap * 100 tweet/nap * 20 szó/tweet)
data_reshaped = data.reshape(-1, TWEET_PER_DAY * MAX_SEQUENCE_LENGTH)
df_data_reshaped = pd.DataFrame(data_reshaped, index=DATES_UNIQUE)


#Load whole date column of preprocessed dataset
FILE_NAME_DATE = 'dataset_prepared_very_date' 
with open(FILE_NAME_DATE + '.pickle', 'rb') as f:
    se_dates = pickle.load(f)
    
#Join daily tweets to Dataset dates
df_dates = se_dates.to_frame()
df_tweet_dataset = df_dates.merge(df_data_reshaped, left_on='REQUESTED_DELIVERY_DATE', right_index=True, how='left')
tweet_dataset = df_tweet_dataset.iloc[:,1:].as_matrix()
tweet_dataset = tweet_dataset.reshape(-1,TWEET_PER_DAY, MAX_SEQUENCE_LENGTH)


'''
###############################################################################
##################          SAVE FOR ENSEMBLE MODEL       #####################
###############################################################################
'''
#Preprocessed tweet dataset
h5f = h5py.File('tweet_dataset.h5', 'w')
h5f.create_dataset('tweet_dataset', data=tweet_dataset)
h5f.close()

#Preprocessed embedding matrix
h5f = h5py.File('wordembedding_matrix.h5', 'w')
h5f.create_dataset('wordembedding_matrix', data=embedding_matrix)
h5f.close()

print('Embedding matrix and tweet vectors are saved to disk')
print('It took {} minutes to process'.format( (time()-START) / 60 ) )


