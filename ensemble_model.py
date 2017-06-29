# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import h5py
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Reshape, Merge, Input, Flatten
from keras.layers.recurrent import LSTM
from keras.layers import Conv1D, Conv2D, MaxPooling1D, MaxPooling2D
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.embeddings import Embedding
from time import time
from scipy.stats import pearsonr
import matemodules.mateutils_byitems as mu
from pandas import read_hdf
import pickle
from datetime import datetime
import os



FORECAST_SHIFT = 7
EPOCHNO = 1
PATIENCE = 7
BATCHSIZE = 128
IS_LOAD_PREV_WEIGHTS = True

'''
###############################################################################
##################          SET NUMERICAL DATA              ###################
###############################################################################
'''
#Numerical part
h5f = h5py.File('generated_for_ensemble/fc_dataset_prepared_for_ensemble.h5','r')
train_x = h5f['train_x'][:]
valid_x = h5f['valid_x'][:]
test_x = h5f['test_x'][:]
train_y = h5f['train_y'][:]
valid_y = h5f['valid_y'][:]
test_y = h5f['test_y'][:]
h5f.close()


embedding_map = {
                 'DAY':         3,
                 'DAY_OF_WEEK': 4,
                 'IS_WEEKEND':  5,
                 'ITEM_ID':     0,
                 'MONTH':       2,
                 'YEAR':        1
                 }



len_uniq_items =         int(train_x[..., embedding_map['ITEM_ID']].max())         + 1 #labelized(0...132) + 1    
len_uniq_years =         int(train_x[..., embedding_map['YEAR']].max())            + 1 #labelized(0...2)   + 1
len_uniq_month =         int(train_x[..., embedding_map['MONTH']].max())           + 1 #labelized(0...11)  + 1
len_uniq_day =           int(train_x[..., embedding_map['DAY']].max())             + 1 #labelized(0...30)  + 1
len_uniq_dayofweek =     int(train_x[..., embedding_map['DAY_OF_WEEK']].max())     + 1 #labelized(0...6)   + 1 
len_uniq_isweekend =     int(train_x[..., embedding_map['IS_WEEKEND']].max())      + 1 #labelized(0...1)   + 1

print('len_uniq_items: ',            len_uniq_items)
print('len_uniq_years: ',            len_uniq_years)
print('len_uniq_month: ',            len_uniq_month)
print('len_uniq_day: ',              len_uniq_day)
print('len_uniq_dayofweek: ',        len_uniq_dayofweek)
print('len_uniq_isweekend: ',        len_uniq_isweekend)



def split_features(X):
    X_list = []
    ls_item =       X[..., [embedding_map['ITEM_ID']]]
    ls_year =       X[..., [embedding_map['YEAR']]]
    ls_month =      X[..., [embedding_map['MONTH']]]
    ls_day =        X[..., [embedding_map['DAY']]]
    ls_dayofweek =  X[..., [embedding_map['DAY_OF_WEEK']]]
    ls_isweekend =  X[..., [embedding_map['IS_WEEKEND']]]
    
    all_other =     X[..., max(embedding_map.values()) + 1:]
    
    
    X_list.append(ls_item)
    X_list.append(ls_year)
    X_list.append(ls_month)
    X_list.append(ls_day)
    X_list.append(ls_dayofweek)
    X_list.append(ls_isweekend)
    X_list.append(all_other)
    return X_list
        
    


#Splitting for embedding
train_x = split_features(train_x)
valid_x = split_features(valid_x) 
test_x = split_features(test_x)




'''
###############################################################################
##################          SET SENTIMENT DATA              ###################
###############################################################################
'''
#Sentiment part
h5f = h5py.File('sentiment/tweet_dataset.h5','r')
tweet_dataset = h5f['tweet_dataset'][:]
h5f.close()

h5f = h5py.File('sentiment/wordembedding_matrix.h5','r')
embedding_matrix = h5f['wordembedding_matrix'][:]
h5f.close()

#XXX:
num_words = 20000
EMBEDDING_DIM = 25
MAX_SEQUENCE_LENGTH = 20
TWEET_PER_DAY = 100


#Split TWEET dataset
len_train = train_x[0].shape[0]
len_valid = valid_x[0].shape[0]
tweet_dataset_train = tweet_dataset[:len_train]
tweet_dataset_valid = tweet_dataset[len_train : len_train + len_valid]
tweet_dataset_test  = tweet_dataset[len_train + len_valid:]

for i in range(TWEET_PER_DAY): 
    train_x.append(tweet_dataset_train[:,i,:].reshape(-1,MAX_SEQUENCE_LENGTH))
    valid_x.append(tweet_dataset_valid[:,i,:].reshape(-1,MAX_SEQUENCE_LENGTH))
    test_x.append(tweet_dataset_test[:,i,:].reshape(-1,MAX_SEQUENCE_LENGTH))
    
    
    


'''
###############################################################################
##################          BUILDING MODEL                  ###################
###############################################################################
'''
print('Building model...')



###############################################################################
##################          NUMERIC MODEL                   ###################
###############################################################################
#Build model
ls_model_numeric = []

model_item = Sequential()
model_item.add(Embedding(len_uniq_items, 27, input_length=1))
model_item.add(Reshape(target_shape=(27,)))
ls_model_numeric.append(model_item)
   
model_year = Sequential()
model_year.add(Embedding(len_uniq_years, 2, input_length=1))
model_year.add(Reshape(target_shape=(2,)))
ls_model_numeric.append(model_year)
   
model_month = Sequential()
model_month.add(Embedding(len_uniq_month, 3, input_length=1))
model_month.add(Reshape(target_shape=(3,)))
ls_model_numeric.append(model_month)

model_day = Sequential()
model_day.add(Embedding(len_uniq_day, 13, input_length=1))
model_day.add(Reshape(target_shape=(13,)))
ls_model_numeric.append(model_day)
   
model_dayofweek = Sequential()
model_dayofweek.add(Embedding(len_uniq_dayofweek, 3, input_length=1))
model_dayofweek.add(Reshape(target_shape=(3,)))
ls_model_numeric.append(model_dayofweek)
   
  
model_isweekend = Sequential()
model_isweekend.add(Embedding(len_uniq_isweekend, 1, input_length=1))
model_isweekend.add(Reshape(target_shape=(1,)))
ls_model_numeric.append(model_isweekend)


model_others = Sequential()
model_others.add(Dense(1, input_dim=train_x[-(TWEET_PER_DAY+1)].shape[1])) #the last is for the tweets
ls_model_numeric.append(model_others)


model_numeric = Sequential()
model_numeric.add(Merge(ls_model_numeric, mode='concat'))
model_numeric.add(Dense(128, init='normal', activation='relu'))
model_numeric.add(Dense(1, init='normal', activation='relu'))



###############################################################################
##################          SENTIMENT MODEL                 ###################
###############################################################################
ls_embeddings = []
for i in range(TWEET_PER_DAY):
    temp_model = Sequential()
    temp_model.add(Embedding(
                            num_words,                              #20000
                            EMBEDDING_DIM,                          #25
                            weights=[embedding_matrix],             #10557*25
                            input_length=MAX_SEQUENCE_LENGTH,       #20
                            trainable=False))    
    ls_embeddings.append(temp_model)                   
    
    
    

#Merge embedding layers
model_sentiment = Sequential()
model_sentiment.add(Merge(ls_embeddings, mode='concat', concat_axis=-2))
print(model_sentiment.layers[-1].output_shape)


#2D
model_sentiment.add(Reshape((TWEET_PER_DAY, MAX_SEQUENCE_LENGTH, EMBEDDING_DIM))) # 100*20*25
print(model_sentiment.layers[-1].output_shape)
model_sentiment.add(Conv2D(256, 1, MAX_SEQUENCE_LENGTH, activation="relu", border_mode='valid'))           
model_sentiment.add(MaxPooling2D(pool_size=(2, 2))) 
model_sentiment.add(Flatten())
model_sentiment.add(Dense(256, activation='relu'))
model_sentiment.add(Dropout(0.4))
model_sentiment.add(Dense(128, activation='relu'))




###############################################################################
##################          ENSEMBLE MODEL                  ###################
###############################################################################
#Ensemble
ensemble_model = Sequential()
ensemble_model.add(Merge([model_numeric, model_sentiment], mode='concat'))
ensemble_model.add(Dense(256, init='normal', activation='relu'))
ensemble_model.add(Dropout(0.3))
ensemble_model.add(Dense(512, init='normal', activation='relu'))
ensemble_model.add(Dropout(0.5))
ensemble_model.add(Dense(1, init='normal', activation='sigmoid'))


'''
###############################################################################
##################          FITTING MODEL                   ###################
###############################################################################
'''
early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1)
checkpointer = ModelCheckpoint(filepath='generated_by_ensemble/best_weights_ensemble.hdf5', verbose=0, save_best_only=True)

ensemble_model.compile(loss='mae', optimizer='adam')
print(ensemble_model.summary())
print('Training is about to start...')
starttime = time()

#Load prev run weights
if IS_LOAD_PREV_WEIGHTS:
    ensemble_model.load_weights('generated_by_ensemble/best_weights_ensemble.hdf5')
    
ensemble_model.fit(
                                train_x,
                                train_y,
                                nb_epoch=EPOCHNO,
                                batch_size=BATCHSIZE,
                                callbacks=[early_stopping, checkpointer],
                                validation_data=(valid_x, valid_y),
                                show_accuracy=True,
                                verbose=1
                            )


ensemble_model.load_weights('generated_by_ensemble/best_weights_ensemble.hdf5')
print('Training time in minutes: ', (time()-starttime) / 60)



'''
###############################################################################
##################          EVALUATE MODEL                  ###################
###############################################################################
'''

#Load Originals
ORIGINAL_PROD_GRP = read_hdf('generated_for_ensemble/saved_originals.h5', 'original_prod')
ORIGINAL_QTY  = read_hdf('generated_for_ensemble/saved_originals.h5', 'original_qty' )
ORIGINAL_DATE  = read_hdf('generated_for_ensemble/saved_originals.h5', 'original_date' )
GROUP_MAXES  = read_hdf('generated_for_ensemble/saved_originals.h5', 'group_maxes' )
print('Originals are loaded from disk')   

#Load Scaler
with open('generated_for_ensemble/test_target_scaler.pickle', 'rb') as f:
    TARGET_SCALER = pickle.load(f)
print('Scaler is loaded from disk') 




#RETURN attr loss (STANDARDIZED form)    
result = ensemble_model.predict(test_x, verbose = 0)
score = ensemble_model.evaluate(test_x, test_y, verbose=0)
print('\n' + 'Test loss: ' + str(score))


#RETURN attr loss (SCALED BACK form)
df_return, figure = mu.scaleback_and_compare_target_prediction(TARGET_SCALER, result, test_y)



#Predicted QTY SOFT METHOD
df_qty_grouped = pd.DataFrame({
                               'DATE'          :  ORIGINAL_DATE,
                               'ITEM_ID'       :  ORIGINAL_PROD_GRP,
                               'QTY_ORIGINAL'  :  ORIGINAL_QTY,
                               'RET_PREDICTED' :  df_return['Prediction_unscaled'].tolist(),
                               'RET_ORIGINAL'  :  df_return['Target_unscaled'].tolist(),
                               })
df_qty_grouped['QTY_PREDICTED'] = np.exp(df_qty_grouped['RET_PREDICTED']) * df_qty_grouped['QTY_ORIGINAL']
df_qty_grouped['QTY_PREDICTED'] = df_qty_grouped.groupby('ITEM_ID')['QTY_PREDICTED'].shift(FORECAST_SHIFT)




'''
###############################################################################
##############       THRESHOLD PREDICTED QUANTITIES          ##################
###############################################################################
'''
#Group by group clipping
def threshold(group):
    max_value = GROUP_MAXES[group.name]
    return group[['QTY_PREDICTED']].clip(0,max_value)   
df_qty_grouped[['QTY_PREDICTED']] = df_qty_grouped.groupby('ITEM_ID')[['QTY_PREDICTED']].apply(threshold)



'''
###############################################################################
##################         CALCULATE LOSSES               #####################
###############################################################################
'''
#Group by group
df_qty_grouped.dropna(how='any', inplace = True)




#SINGLE VALUES
results = {}
results['CORR'] = pearsonr(df_qty_grouped['QTY_ORIGINAL'], df_qty_grouped['QTY_PREDICTED'])[0]
results['MAE'] = np.mean(abs(df_qty_grouped['QTY_ORIGINAL'] - df_qty_grouped['QTY_PREDICTED']))
results['RMSE'] = np.sqrt(np.mean((df_qty_grouped['QTY_ORIGINAL'] - df_qty_grouped['QTY_PREDICTED']) ** 2)) #mean not good for RMSE
results['MAPE'] = np.mean(abs((df_qty_grouped['QTY_ORIGINAL'] - df_qty_grouped['QTY_PREDICTED']) / df_qty_grouped['QTY_ORIGINAL'])) * 100

#Calculating MASE
import h5py
h5f = h5py.File('results_final/ORIGINAL_TRAIN_Y.h5','r')
ORIGINAL_TRAIN_Y = h5f['ORIGINAL_TRAIN_Y'][:]
h5f.close()
ORIGINAL_TRAIN_Y = pd.Series(ORIGINAL_TRAIN_Y)
n = ORIGINAL_TRAIN_Y.shape[0]
d = abs(ORIGINAL_TRAIN_Y.diff()).sum() / (n-1)
errors = np.abs(df_qty_grouped['QTY_ORIGINAL'] - df_qty_grouped['QTY_PREDICTED'])
results['MASE'] = errors.mean() / d



#Display results
print('CORR_QTY', results['CORR'])
print('MAE_QTY', results['MAE'])
print('RMSE_QTY', results['RMSE'])
print('MAPE: ', results['MAPE'])
print('MASE: ', results['MASE'])


#FIXME: temp
print(df_qty_grouped[['QTY_ORIGINAL','QTY_PREDICTED']][:100])

#Create unique folder
unique_folder = 'losses_{:06.4f}_{}_{}_{}'.format(results['MASE'], int(results['MAE']), int(results['RMSE']), int(results['MAPE']))
path_folder = os.path.join('generated_by_ensemble', unique_folder)
os.mkdir(path_folder)

path_losses = os.path.join(path_folder, 'losses.h5')
path_predictions = os.path.join(path_folder, 'predictions.h5')

#Save losses and predictions
df_results = pd.DataFrame(results, index=[0])
df_results.to_hdf(path_losses, 'losses', mode='w')
df_qty_grouped.to_hdf(path_predictions, 'predictions', mode='w')


#Save source code
path_source_code = os.path.join(path_folder, 'source_code.txt')
with open(__file__) as sc:
    source_code = sc.read()
    
with open(path_source_code, 'w') as text_file:
    text_file.write(source_code)


print('Results are saved to disk')

