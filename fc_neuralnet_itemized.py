# -*- coding: utf-8 -*-


'''
###############################################################################
##################          IMPORTS             ###############################
###############################################################################
'''
from keras.models import Sequential
from keras.layers import Dense, Dropout, Reshape, Merge
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from keras.layers.embeddings import Embedding
from sklearn.preprocessing import LabelEncoder
from sklearn import manifold
import matplotlib.pyplot as plt
from os import getcwd
from os.path import dirname
import sys
if dirname(getcwd()) not in sys.path:       #if it doesn't contain yet
    sys.path.insert(0, getcwd())            #add parent folder to path
import matemodules.mateutils_byitems as mu
import pandas as pd
from pandas import HDFStore
import numpy as np
import pickle
import gc
from scipy.stats import pearsonr
import time
from datetime import datetime
import os

'''
###############################################################################
##################          GENERAL SETTINGS          #########################
###############################################################################
'''
gc.collect()
SIZE = 20
plt.rc('font', size=SIZE)        # controls default text sizes
plt.rc('axes', titlesize=30)   # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)   # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)  # legend fontsize
plt.rc('figure', titlesize=SIZE)    
plt.rcParams['figure.figsize'] = 16, 12
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.markersize'] = 10
plt.style.use('ggplot')


'''
###############################################################################
##################          SWITCHES            ###############################
###############################################################################
MODEL_TYPE can be: FC_NO_EE/FC_WITH_EE/TEST_FEATURE_GENERATION
'''

FORECAST_SHIFT = 7
TESTSET_SIZE = 20

#MODEL_TYPE = 'FC_NO_EE'
MODEL_TYPE = 'FC_WITH_EE'
#MODEL_TYPE = 'TEST_FEATURE_GENERATION'

ENSEMBLE_AND_HYPEROPT_PREPARATION = True
SAVE_MOVEMENT_FOR_SENTIMENT = False

BATCHSIZE = 1024
EPOCHNO = 2
PATIENCE = 10
TARGET_FEATURE = 'RETURN'
ROLLIN_WINDOWS = 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15


'''
###############################################################################
##################          LOAD FROM DISK        #############################
###############################################################################
'''
#Load data
with open('generated_data/data_prepared_itemized.pickle', 'rb') as f:
    dataset = pickle.load(f)
print('Dataset is loaded from disk')


'''
###############################################################################
##################          ATTR CREATION       ###############################
###############################################################################
'''
#Call the column generating functions
mu.add_rollin_mean_cols(dataset, 'ITEM_ID', 'REQUESTED_QUANTITY', *ROLLIN_WINDOWS)


#Drop NaNs
dataset.dropna(inplace=True)
dataset.reset_index(inplace=True, drop=True)



'''
###############################################################################
##################          ENCODING CATEGORY LABELS           ################
###############################################################################
'''
#String to labels (0,1,2,3..)
if MODEL_TYPE == 'FC_WITH_EE':
    labelencoder = LabelEncoder()
    dataset['ITEM_ID'] = labelencoder.fit_transform(dataset['ITEM_ID']) 
    dataset['YEAR'] = LabelEncoder().fit_transform(dataset['YEAR']) 
    dataset['MONTH'] = LabelEncoder().fit_transform(dataset['MONTH']) 
    dataset['DAY'] = LabelEncoder().fit_transform(dataset['DAY']) 
    dataset['DAY_OF_WEEK'] = LabelEncoder().fit_transform(dataset['DAY_OF_WEEK']) 
    dataset['IS_WEEKEND'] = LabelEncoder().fit_transform(dataset['IS_WEEKEND']) 


elif MODEL_TYPE == 'FC_NO_EE' or MODEL_TYPE == 'TEST_FEATURE_GENERATION':
    one_hot = pd.get_dummies(dataset['ITEM_ID'], prefix='ITEM_ID')    #One-hot encode
    dataset = dataset.join(one_hot)
    del one_hot
    

'''
###############################################################################
##################          ATTR NAMES           ##############################
###############################################################################
'''
#Collect columns by name
col_names = dataset.columns
itemid_cols =  [name for name in col_names if name.startswith('ITEM_ID_') ] #It will be empty in case of EntityEmbedding (cool)
movin_mean_cols = [name for name in col_names if name.startswith('ROLLING_MEAN') ]

                    
output_names = [TARGET_FEATURE]                #ln(qty_i+x/qty_i)
input_names = [name for name in col_names if name != TARGET_FEATURE]
print('Features are created')



'''
###############################################################################
##################          SPLITTING           ###############################
###############################################################################
'''

#Split to train, valid, tst
VERY_ORIGINAL_DATE = dataset['REQUESTED_DELIVERY_DATE']
train_set = dataset.groupby('ITEM_ID').apply(lambda x: x[:-(2*TESTSET_SIZE)])
valid_set = dataset.groupby('ITEM_ID').apply(lambda x: x[-(2*TESTSET_SIZE) : -TESTSET_SIZE])
test_set = dataset.groupby('ITEM_ID').apply(lambda x: x[-TESTSET_SIZE:])
del dataset
train_set.reset_index(inplace=True, drop=True)
valid_set.reset_index(inplace=True, drop=True)
test_set.reset_index(inplace=True, drop=True)



#For evaluation save originals
ORIGINAL_PROD_GRP = test_set['ITEM_ID'].copy()
ORIGINAL_QTY = test_set['REQUESTED_QUANTITY'].copy()
ORIGINAL_DATE = test_set['REQUESTED_DELIVERY_DATE'].copy()
#Save for MASE loss metric
ORIGINAL_TRAIN_Y = train_set['REQUESTED_QUANTITY'].copy()

#Save goup maxes for thresholding (Postprocessing)
GROUP_MAXES = train_set.groupby('ITEM_ID')['REQUESTED_QUANTITY'].max()


train_x = train_set[input_names]
train_y = train_set[output_names] 
del train_set
valid_x = valid_set[input_names]
valid_y = valid_set[output_names]
del valid_set
test_x = test_set[input_names]
test_y = test_set[output_names]
del test_set
print('Splitting has happened')



'''
###############################################################################
##################          STANDARDIZATION           #########################
###############################################################################
'''

if MODEL_TYPE == 'TEST_FEATURE_GENERATION': #For generating all inputs for testing
    minmax_names = [
                           'YEAR', 
                           'MONTH',
                           'DAY',
                           'DAY_OF_WEEK',
                           'IS_WEEKEND',
                           'NEXT_HOLIDAY',
                           'PREV_HOLIDAY'
                           ] + itemid_cols

    standard_names = movin_mean_cols
    noscale_names = ['ITEM_ID', 'REQUESTED_QUANTITY']
                           

                           
elif MODEL_TYPE == 'FC_NO_EE' or MODEL_TYPE == 'FC_WITH_EE':   #For FC net (EE and Without EE)                              
    minmax_names = [        
                           'YEAR', 
                           'MONTH',
                           'DAY',
                           'DAY_OF_WEEK',
                           'IS_WEEKEND',
                           'NEXT_HOLIDAY',
                           'PREV_HOLIDAY'
                           ] + itemid_cols
    
    standard_names = movin_mean_cols + ['ZEROS_CUMSUM', 'ZERO_FULL_RATIO', 'MAX_ZERO_SEQUENCE', 'MEAN_OF_ZERO_SEQ', 'ZERO_QTY_SEQUENCE', 'ORDERS_THAT_DAY']
    noscale_names = []

   


if MODEL_TYPE == 'FC_WITH_EE':
    #Embedding needs this
    embedding_features = ['ITEM_ID','YEAR','MONTH','DAY','DAY_OF_WEEK','IS_WEEKEND']
    noscale_names += embedding_features
    minmax_names = [name for name in minmax_names if name not in embedding_features]


#For all
print('Standard names: ', standard_names)    
print('Minmax names: ', minmax_names) 
    

spec_standard = mu.SpecificStandardization()
spec_standard.set_features_to_minmax_scale(minmax_names)
spec_standard.set_features_to_standard_scale(standard_names)
spec_standard.set_features_to_no_scale(noscale_names)

train_x,valid_x,test_x,train_y,valid_y,test_y = spec_standard.standardize_all_sepcificly(train_x,valid_x,test_x,train_y,valid_y,test_y, is_it_lstm=False)


#Get column index of noscale columns
noscale_index = {name:test_x.columns.get_loc(name) for name in noscale_names}

'''
###############################################################################
##################          SENTIMENT MODEL DATASET       #####################
###############################################################################
'''
if SAVE_MOVEMENT_FOR_SENTIMENT:
    import h5py
    h5f = h5py.File('generated_for_sentiment/movements.h5', 'w')
    h5f.create_dataset('train_movement', data=train_x['MOVEMENT'])
    h5f.create_dataset('valid_movement', data=valid_x['MOVEMENT'])
    h5f.create_dataset('test_movement', data=test_x['MOVEMENT'])
    h5f.close()

    raise SystemExit

if MODEL_TYPE == 'TEST_FEATURE_GENERATION': 
    '''
    ###############################################################################
    ##################          SAVE TEST DATA AND QUIT       #####################
    ###############################################################################
    '''
    hdf = HDFStore('test_inputs/source_data/test_data_all_features_itemized.h5')
    hdf.put('train_x',  train_x )
    hdf.put('valid_x',  valid_x )
    hdf.put('test_x' ,  test_x  )
    hdf.put('train_y',  train_y )
    hdf.put('valid_y',  valid_y )
    hdf.put('test_y' ,  test_y  )
    hdf.close()
    print('Data set is generated and saved')
    
    hdf = HDFStore('test_inputs/source_data/saved_originals_itemized.h5')
    hdf.put('original_prod',  ORIGINAL_PROD_GRP )
    hdf.put('original_qty',   ORIGINAL_QTY )
    hdf.put('original_date',  ORIGINAL_DATE )
    hdf.put('group_maxes',    GROUP_MAXES )
    hdf.close()
    print('Original prod names and quantities are saved')
    

    with open('test_inputs/source_data/test_target_scaler_itemized.pickle', 'wb') as f:
        pickle.dump(spec_standard.get_scaler('target'), f, -1)
    print('Scaler object is saved')

    raise SystemExit



train_x = train_x.as_matrix()
valid_x = valid_x.as_matrix()
test_x  = test_x.as_matrix()
train_y = train_y.as_matrix()
valid_y = valid_y.as_matrix()
test_y  = test_y.as_matrix()
print('Standardization was successful')



'''
###############################################################################
##################          SAVE FOR ENSMEBLE MODEL     ######################
###############################################################################
'''
if ENSEMBLE_AND_HYPEROPT_PREPARATION:
    import h5py
    h5f = h5py.File('generated_for_ensemble/fc_dataset_prepared_for_ensemble.h5', 'w')
    h5f.create_dataset('train_x', data=train_x)
    h5f.create_dataset('valid_x', data=valid_x)
    h5f.create_dataset('test_x', data=test_x)
    h5f.create_dataset('train_y', data=train_y)
    h5f.create_dataset('valid_y', data=valid_y)
    h5f.create_dataset('test_y', data=test_y)
    h5f.close()

    with open('sentiment/dataset_prepared_very_date.pickle', 'wb') as f:
        pickle.dump(VERY_ORIGINAL_DATE, f, 2)
        

    hdf = HDFStore('generated_for_ensemble/saved_originals.h5')
    hdf.put('original_prod',  ORIGINAL_PROD_GRP )
    hdf.put('original_qty',   ORIGINAL_QTY )
    hdf.put('original_date',  ORIGINAL_DATE )
    hdf.put('group_maxes',    GROUP_MAXES )
    hdf.close()
    print('Original prod names and quantities are saved')
    

    with open('generated_for_ensemble/test_target_scaler.pickle', 'wb') as f:
        pickle.dump(spec_standard.get_scaler('target'), f, -1)
    print('Scaler object is saved')

    
    print('Dataset and ORIGINALS are saved for ensemble mmodel')
    raise SystemExit


'''
###############################################################################
##################          TRAINING MODEL           ##########################
###############################################################################
'''
#Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1)
checkpointer = ModelCheckpoint(filepath='generated_by_fc_itemized/best_weights_itemized.hdf5', verbose=0, save_best_only=True)
print('Building model...')

if MODEL_TYPE == 'FC_NO_EE':
    model = Sequential()   
    model.add(Dense(128, input_dim=train_x.shape[1], init='normal', activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(128, init='normal', activation='relu'))
    model.add(Dense(1, init='normal', activation='sigmoid'))



elif MODEL_TYPE == 'FC_WITH_EE':
    len_uniq_items =         int(train_x[..., noscale_index['ITEM_ID']].max())         + 1 #labelized(0...132) + 1    
    len_uniq_years =         int(train_x[..., noscale_index['YEAR']].max())            + 1 #labelized(0...2)   + 1
    len_uniq_month =         int(train_x[..., noscale_index['MONTH']].max())           + 1 #labelized(0...11)  + 1
    len_uniq_day =           int(train_x[..., noscale_index['DAY']].max())             + 1 #labelized(0...30)  + 1
    len_uniq_dayofweek =     int(train_x[..., noscale_index['DAY_OF_WEEK']].max())     + 1 #labelized(0...6)   + 1 
    len_uniq_isweekend =     int(train_x[..., noscale_index['IS_WEEKEND']].max())      + 1 #labelized(0...1)   + 1
    
    print('len_uniq_items: ',            len_uniq_items)
    print('len_uniq_years: ',            len_uniq_years)
    print('len_uniq_month: ',            len_uniq_month)
    print('len_uniq_day: ',              len_uniq_day)
    print('len_uniq_dayofweek: ',        len_uniq_dayofweek)
    print('len_uniq_isweekend: ',        len_uniq_isweekend)
    

    
    def split_features(X):
        X_list = []
        ls_item =       X[..., [noscale_index['ITEM_ID']]]
        ls_year =       X[..., [noscale_index['YEAR']]]
        ls_month =      X[..., [noscale_index['MONTH']]]
        ls_day =        X[..., [noscale_index['DAY']]]
        ls_dayofweek =  X[..., [noscale_index['DAY_OF_WEEK']]]
        ls_isweekend =  X[..., [noscale_index['IS_WEEKEND']]]
        
        all_other =     X[..., max(noscale_index.values()) + 1:]
        
        
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
    
    
    #Build model
    models = []

    
    model_item = Sequential()
    model_item.add(Embedding(len_uniq_items, 27, input_length=1))
    model_item.add(Reshape(target_shape=(27,)))
    models.append(model_item)
   
    model_year = Sequential()
    model_year.add(Embedding(len_uniq_years, 2, input_length=1))
    model_year.add(Reshape(target_shape=(2,)))
    models.append(model_year)
   
    model_month = Sequential()
    model_month.add(Embedding(len_uniq_month, 3, input_length=1))
    model_month.add(Reshape(target_shape=(3,)))
    models.append(model_month)
    
    model_day = Sequential()
    model_day.add(Embedding(len_uniq_day, 13, input_length=1))
    model_day.add(Reshape(target_shape=(13,)))
    models.append(model_day)
   
    model_dayofweek = Sequential()
    model_dayofweek.add(Embedding(len_uniq_dayofweek, 3, input_length=1))
    model_dayofweek.add(Reshape(target_shape=(3,)))
    models.append(model_dayofweek)
   
  
    model_isweekend = Sequential()
    model_isweekend.add(Embedding(len_uniq_isweekend, 1, input_length=1))
    model_isweekend.add(Reshape(target_shape=(1,)))
    models.append(model_isweekend)
  

    
    model_others = Sequential()
    model_others.add(Dense(1, input_dim=train_x[-1].shape[1]))
    models.append(model_others)
    model = Sequential()
    model.add(Merge(models, mode='concat'))
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(512, init='normal', activation='relu'))
    model.add(Dropout(0.1))
    model.add(Dense(1, init='normal', activation='sigmoid'))
    
    


model.compile(loss='mae', optimizer='adam')
print(model.summary())
print('Training is about to start...')
starttime = time.time()
history = model.fit(train_x,
                    train_y,
                    nb_epoch=EPOCHNO,
                    batch_size=BATCHSIZE,
                    callbacks=[early_stopping,checkpointer],
                    validation_data=(valid_x, valid_y),
                    verbose=1)

model.load_weights('generated_by_fc_itemized/best_weights_itemized.hdf5')
print('Training time in minutes: ', (time.time()-starttime) / 60)



'''
###############################################################################
##################          EVALUATING RETURN VALUES     ######################
###############################################################################
'''
print('Evaluating model...')

#Display the LEARNING CURVE
compare_loss = pd.DataFrame({'Tanító halmaz hibagörbéje': history.history['loss'], 'Validációs halmaz hibagörbéje': history.history['val_loss']})
plt.rc('axes', titlesize=30) 
plt.rc('legend', fontsize=25)
ax = compare_loss.plot(figsize=(16, 12))
ax.set_xlabel(u"Epoch szám")
ax.set_ylabel(u"MAE hiba érték")
ax.set_title(u'FC hálózat hibájának alakulása tanítás közben', y=1.05)
plt.tight_layout()


#RETURN attr loss (STANDARDIZED form)    
result = model.predict(test_x, verbose = 0)
score = model.evaluate(test_x, test_y, verbose=0)
print('\n' + 'Test loss: ' + str(score))


#RETURN attr loss (SCALED BACK form)
df_return, figure = mu.scaleback_and_compare_target_prediction(spec_standard.get_scaler('target'), result, test_y)
figure.show()
mse, corr_ret, result_describe, target_describe = mu.compare_target_prediction(df_return['Prediction_unscaled'].tolist(),df_return['Target_unscaled'].tolist())

mse_of_originalscale_return = np.mean( (df_return['Prediction_unscaled'] - df_return['Target_unscaled']) ** 2) 
print('Test RETURN MSE (original scale): ', mse_of_originalscale_return)



'''
###############################################################################
##################          CALCULATE SOFT QUANTITY       #####################
###############################################################################
'''
#return_i = log(qty_i+1/qty_i)
#exp(return_i) = qty_i+1/qty_i
#qty_i+1 = exp(return_i) * qty_i


#Delabelize
if MODEL_TYPE == 'FC_WITH_EE':
    ORIGINAL_PROD_GRP = labelencoder.inverse_transform(ORIGINAL_PROD_GRP)
    GROUP_MAXES.index = labelencoder.inverse_transform(GROUP_MAXES.index)



#Predicted QTY
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
results['CORR_RET'] = corr_ret[0]
results['CORR'] = pearsonr(df_qty_grouped['QTY_ORIGINAL'], df_qty_grouped['QTY_PREDICTED'])[0]
results['MAE'] = np.mean(abs(df_qty_grouped['QTY_ORIGINAL'] - df_qty_grouped['QTY_PREDICTED']))
results['RMSE'] = np.sqrt(np.mean((df_qty_grouped['QTY_ORIGINAL'] - df_qty_grouped['QTY_PREDICTED']) ** 2)) #mean not good for RMSE
results['MAPE'] = np.mean(abs((df_qty_grouped['QTY_ORIGINAL'] - df_qty_grouped['QTY_PREDICTED']) / df_qty_grouped['QTY_ORIGINAL'])) * 100

#Calculating MASE
n = ORIGINAL_TRAIN_Y.shape[0]
d = abs(ORIGINAL_TRAIN_Y.diff()).sum() / (n-1)
errors = np.abs(df_qty_grouped['QTY_ORIGINAL'] - df_qty_grouped['QTY_PREDICTED'])
results['MASE'] = errors.mean() / d



#Display results
print('CORR_RET', results['CORR_RET'])
print('CORR_QTY', results['CORR'])
print('MAE_QTY', results['MAE'])
print('RMSE_QTY', results['RMSE'])
print('MAPE: ', results['MAPE'])
print('MASE', results['MASE'])

'''
###############################################################################
##################          SAVE RESULTS                      #################
###############################################################################
'''
unique_folder = 'losses_{:06.4f}_{:06.4f}_{}_{}_{}'.format(results['MASE'], mse_of_originalscale_return, int(results['MAE']), int(results['RMSE']), int(results['MAPE']))


path_folder = os.path.join('generated_by_fc_itemized', unique_folder)
os.mkdir(path_folder)

path_losses = os.path.join(path_folder, 'losses.h5')
path_predictions = os.path.join(path_folder, 'predictions.h5')

#Save losses and predictions
df_results = pd.DataFrame(results, index=[0])
df_results.to_hdf(path_losses, 'losses', mode='w')
df_qty_grouped.to_hdf(path_predictions, 'predictions', mode='w')
print('Results are saved to disk')


#Save source code
path_source_code = os.path.join(path_folder, 'source_code.txt')
with open(__file__) as sc:
    source_code = sc.read()
    
with open(path_source_code, 'w') as text_file:
    text_file.write(source_code)



'''
###############################################################################
##################          DISPLAY FORECAST CURVES           #################
###############################################################################
'''
df_qty_grouped.reset_index(inplace=True, drop=True)
df_qty_grouped[['QTY_ORIGINAL','QTY_PREDICTED']].iloc[1000:1300].plot()


                     
          
        
        
