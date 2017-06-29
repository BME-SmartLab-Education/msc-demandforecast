# -*- coding: utf-8 -*-


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.recurrent import LSTM
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
from os import getcwd
from os.path import dirname
import sys
if dirname(getcwd()) not in sys.path:     #if it doesn't contain yet
    sys.path.insert(0, getcwd())          #add parent folder to path
import matemodules.mateutils_byitems as mu
import pandas as pd
import numpy as np
import pickle
import gc
from scipy.stats import pearsonr
import time
import h5py
from datetime import datetime
import os


'''
###############################################################################
##################          GENERAL SETTINGS          #########################
###############################################################################
'''
gc.collect()
SIZE = 20
plt.rc('axes', titlesize=SIZE)   # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)   # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)  # legend fontsize
plt.rc('figure', titlesize=SIZE)    
plt.rcParams['figure.figsize'] = 16, 12
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.markersize'] = 10
plt.style.use('ggplot')
plt.rc('font', size=30)          # controls default text sizes


'''
###############################################################################
##################          SWITCHES            ###############################
###############################################################################
'''

FORECAST_SHIFT = 7
TESTSET_SIZE = 20


#RUNNING_MODE = 'TEST_FEATURE_GENERATION'
#RUNNING_MODE = 'LSTM_RUN'
RUNNING_MODE = '1D_CONVNET'
TARGET_FEATURE = 'RETURN'
ROLLIN_WINDOWS = 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15
ENSEMBLE_AND_HYPEROPT_PREPARATION = False

#Train params
EPOCHNO = 100
BATCHSIZE = 1024
PATIENCE = 10



'''
###############################################################################
##################          LOAD FROM DISK        #############################
###############################################################################
'''
#Load data
with open('generated_data/data_prepared_itemized.pickle', 'rb') as f:
    dataset = pickle.load(f)
print('Dataset is loaded from disk')

with open('generated_by_fc_itemized/embedding_weights_itemized.pickle', 'rb') as f:
    embeddings = pickle.load(f)
print('Embedding data is loaded from disk')

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
##################          LABELIZING FEATURES           #####################
###############################################################################
'''
label_encoder = LabelEncoder()
dataset['ITEM_ID'] = label_encoder.fit_transform(dataset['ITEM_ID']) 
dataset['YEAR'] = LabelEncoder().fit_transform(dataset['YEAR']) 
dataset['MONTH'] = LabelEncoder().fit_transform(dataset['MONTH']) 
dataset['DAY'] = LabelEncoder().fit_transform(dataset['DAY']) 
dataset['DAY_OF_WEEK'] = LabelEncoder().fit_transform(dataset['DAY_OF_WEEK']) 
dataset['IS_WEEKEND'] = LabelEncoder().fit_transform(dataset['IS_WEEKEND']) 




'''
###############################################################################
##################          FEATURE SELECTION           #######################
###############################################################################
'''
#Collect columns by name
col_names = dataset.columns
itemid_cols =  [name for name in col_names if name.startswith('ITEM_ID_') ]        #It will be empty in case of EntityEmbedding
movin_mean_cols = [name for name in col_names if name.startswith('ROLLING_MEAN') ]

#Set specific standardization for features
if RUNNING_MODE == 'TEST_FEATURE_GENERATION': #For generating all inputs for testing
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

    
    
elif RUNNING_MODE == 'LSTM_RUN' or RUNNING_MODE == '1D_CONVNET': 
  input_names = [                
                       'ITEM_ID',
                       'YEAR',
                       'MONTH',
                       'DAY',
                       'DAY_OF_WEEK',
                       'IS_WEEKEND',
                       'NEXT_HOLIDAY',
                       'PREV_HOLIDAY'
                       ] + movin_mean_cols + ['ZEROS_CUMSUM', 'ZERO_FULL_RATIO', 'MAX_ZERO_SEQUENCE', 'MEAN_OF_ZERO_SEQ', 'ZERO_QTY_SEQUENCE', 'ORDERS_THAT_DAY']

output_names = [TARGET_FEATURE]
original_names = ['REQUESTED_QUANTITY','REQUESTED_DELIVERY_DATE']
dataset = dataset[input_names + output_names + original_names]
print('Features are selected')
print('Input names: ', input_names)



'''
###############################################################################
##################          SPLITTING           ###############################
###############################################################################
'''
#Split to train, valid, tst
train_set = dataset.groupby('ITEM_ID').apply(lambda x: x[:-(2*TESTSET_SIZE)])
valid_set = dataset.groupby('ITEM_ID').apply(lambda x: x[-(2*TESTSET_SIZE) : -TESTSET_SIZE])
test_set = dataset.groupby('ITEM_ID').apply(lambda x: x[-TESTSET_SIZE:])
train_set.reset_index(inplace=True, drop=True)
valid_set.reset_index(inplace=True, drop=True)
test_set.reset_index(inplace=True, drop=True)
del dataset


#Save originals (1+3)
GROUP_MAXES = train_set.groupby('ITEM_ID')['REQUESTED_QUANTITY'].max()

ORIGINAL_PROD_GRP = {}
ORIGINAL_QTY = {}
ORIGINAL_DATE = {}
ORIGINAL_PROD_GRP['train'] = train_set['ITEM_ID']
ORIGINAL_PROD_GRP['valid'] = valid_set['ITEM_ID']
ORIGINAL_PROD_GRP['test'] = test_set['ITEM_ID']
ORIGINAL_QTY['train'] = train_set['REQUESTED_QUANTITY']
ORIGINAL_QTY['valid'] = valid_set['REQUESTED_QUANTITY']
ORIGINAL_QTY['test'] = test_set['REQUESTED_QUANTITY']
ORIGINAL_DATE['train'] = train_set['REQUESTED_DELIVERY_DATE']
ORIGINAL_DATE['valid'] = valid_set['REQUESTED_DELIVERY_DATE']
ORIGINAL_DATE['test'] = test_set['REQUESTED_DELIVERY_DATE']



#Save for concatenation
temp_train = train_set[original_names].copy()
temp_valid = valid_set[original_names].copy()
temp_test = test_set[original_names].copy()


#Drop the original temp cols
train_set = train_set[input_names + output_names]
valid_set = valid_set[input_names + output_names]
test_set = test_set[input_names + output_names]

print('Splitting has happened')




'''
###############################################################################
##################          COLUMN MAGIC                  #####################
###############################################################################
'''

#Embedding name : embedding dim
embedding_dim_map = {
                            'ITEM_ID'       : embeddings[0].shape[1],
                            'YEAR'          : embeddings[1].shape[1],
                            'MONTH'         : embeddings[2].shape[1],
                            'DAY'           : embeddings[3].shape[1],
                            'DAY_OF_WEEK'   : embeddings[4].shape[1],
                            'IS_WEEKEND'    : embeddings[5].shape[1]
                        }


#Generate column names
def generate_embedding_names(col_name):
    generated_names = []
    for i in range(embedding_dim_map[col_name]):
        new_name = col_name + '_' + str(i)
        generated_names += [new_name]
    return generated_names
    
    
#Get the column names
columns = []
for col in train_set.columns:
    if col in embedding_dim_map.keys():
        actual_col = generate_embedding_names(col)
    else:
        actual_col = [col]
        
    columns += actual_col
            





'''
###############################################################################
##################          USE EMBEDDING WEIGHTS         #####################
###############################################################################
'''
#Embedding index map
index_embedding_mapping = {
                            train_set.columns.get_loc('ITEM_ID'):       0,
                            train_set.columns.get_loc('YEAR'):          1,
                            train_set.columns.get_loc('MONTH'):         2,
                            train_set.columns.get_loc('DAY'):           3,
                            train_set.columns.get_loc('DAY_OF_WEEK'):   4,
                            train_set.columns.get_loc('IS_WEEKEND'):    5
                            }


#Load embedding features as inputs
def embedding_input(data):                                
    X = data.as_matrix()                            
    X_embedded = []
    
    for record in X:
        embedded_features = []
        for i, feat in enumerate(record):
            if i not in index_embedding_mapping.keys():
                embedded_features += [feat]
            else:
                feat = int(feat)
                embedding_index = index_embedding_mapping[i]
                embedded_features += embeddings[embedding_index][feat].tolist()
    
        X_embedded.append(embedded_features)
    
    return np.array(X_embedded)

train_set = embedding_input(train_set)
valid_set = embedding_input(valid_set)
test_set = embedding_input(test_set)
print('Embedding weight are in use')




'''
###############################################################################
##################          STANDARDIZATION           #########################
###############################################################################
'''
#Standardize
scaler_standard = preprocessing.StandardScaler().fit(train_set[:,:-1])
train_set[:,:-1] = scaler_standard.transform(train_set[:,:-1])
valid_set[:,:-1] = scaler_standard.transform(valid_set[:,:-1])
test_set[:,:-1] = scaler_standard.transform(test_set[:,:-1])

#Normalize
target_scaler = preprocessing.MinMaxScaler().fit(train_set[:,-1])
train_set[:,-1] = target_scaler.transform(train_set[:,-1])
valid_set[:,-1] = target_scaler.transform(valid_set[:,-1])
test_set[:,-1] = target_scaler.transform(test_set[:,-1])
print('Data is standardized')


'''
###############################################################################
##################          RESTORING DATAFRAME           #####################
###############################################################################
'''
#Transform to dataframe
train_set = pd.DataFrame(train_set, columns=columns)
valid_set = pd.DataFrame(valid_set, columns=columns)
test_set = pd.DataFrame(test_set, columns=columns)


#Concatenate by columns
train_set = pd.concat([train_set, ORIGINAL_PROD_GRP['train'], ORIGINAL_QTY['train'], ORIGINAL_DATE['train']], axis=1)
valid_set = pd.concat([valid_set, ORIGINAL_PROD_GRP['valid'], ORIGINAL_QTY['valid'], ORIGINAL_DATE['valid']], axis=1)
test_set = pd.concat([test_set, ORIGINAL_PROD_GRP['test'], ORIGINAL_QTY['test'], ORIGINAL_DATE['test']], axis=1)



'''
###############################################################################
##################          CREATING LSTM BATCHES          ####################
###############################################################################
'''
#############     LSTM PARAMS             ##################
input_names = [col for col in train_set.columns if col not in [TARGET_FEATURE, 'ITEM_ID', 'REQUESTED_QUANTITY', 'REQUESTED_DELIVERY_DATE']]
sample_points = 10
shift = 1


#############     TRAINING DATA    ########################
ins = []
outs = []
cntr = 0
for name, group in train_set.groupby('ITEM_ID'): 
    cntr += 1
    print('Name: ', name)
    print('Round: ', cntr)
    nb_samples = len(group) - sample_points + 1
    input_list = [[group[input_names].iloc[i:i+sample_points].as_matrix()] for i in range(0,nb_samples,shift)]
    target_list = [np.atleast_2d(group[TARGET_FEATURE].iloc[i+sample_points-1]) for i in range(0,nb_samples,shift)]
    ins.append(np.concatenate(input_list,axis=0))
    outs.append(np.concatenate(target_list,axis = 0))
      
        
train_x = np.vstack(ins)
train_x = np.array(train_x, dtype=np.float)
train_y = np.vstack(outs)


#############     VALID DATA    ########################
test_start = valid_set.groupby('ITEM_ID').apply(lambda x: x[-(sample_points-1):]).copy()


ins = []
outs = []
cntr = 0
for name, group in valid_set.groupby('ITEM_ID'): 
    cntr += 1
    print('Name: ', name)
    print('Round: ', cntr)
    nb_samples = len(group) - sample_points + 1
    input_list = [[group[input_names].iloc[i:i+sample_points].as_matrix()] for i in range(0,nb_samples,shift)]
    target_list = [np.atleast_2d(group[TARGET_FEATURE].iloc[i+sample_points-1]) for i in range(0,nb_samples,shift)] 
    ins.append(np.concatenate(input_list,axis=0))
    outs.append(np.concatenate(target_list,axis = 0))

valid_x = np.vstack(ins)
valid_x = np.array(valid_x, dtype=np.float)
valid_y = np.vstack(outs)



#############     TEST DATA    ########################
test_set = test_set.append(test_start)
test_set.sort_values(by=['ITEM_ID','REQUESTED_DELIVERY_DATE'],inplace=True)


ins = []
outs = []
originals_test = []
cntr = 0
for name, group in test_set.groupby('ITEM_ID'): 
    cntr += 1
    print('Name: ', name)
    print('Round: ', cntr)
    nb_samples = len(group) - sample_points + 1
    input_list = [[group[input_names].iloc[i:i+sample_points].as_matrix()] for i in range(0,nb_samples,shift)]
    target_list = [np.atleast_2d(group[TARGET_FEATURE].iloc[i+sample_points-1]) for i in range(0,nb_samples,shift)]
    ins.append(np.concatenate(input_list, axis=0))
    outs.append(np.concatenate(target_list, axis=0))


test_x = np.vstack(ins)
test_x = np.array(test_x, dtype=np.float)
test_y = np.vstack(outs)





#if RUNNING_MODE == 'TEST_FEATURE_GENERATION':   
#    h5f = h5py.File('test_inputs/source_data/lstm_test_data_all_features.h5', 'w')
#    h5f.create_dataset('train_x',  data=train_x)
#    h5f.create_dataset('valid_x',  data=valid_x)
#    h5f.create_dataset('test_x' ,  data=test_x)
#    h5f.create_dataset('train_y',  data=train_y)
#    h5f.create_dataset('valid_y',  data=valid_y)
#    h5f.create_dataset('test_y' ,  data=test_y)
#    h5f.close()
#    print('Data set is generated and saved')
#    
#
#    with open('test_inputs/source_data/lstm_saved_originals.pickle', 'wb') as f:
#        pickle.dump([ORIGINAL_COLUMNS, ORIGINAL_PROD_GRP, ORIGINAL_QTY, ORIGINAL_DATE, GROUP_MAXES], f, -1)
#    print('Original values are saved')
#    
#
#    with open('test_inputs/source_data/lstm_test_target_scaler.pickle', 'wb') as f:
#        pickle.dump(spec_standard.get_scaler('target'), f, -1)
#    print('Scaler object is saved')
#    
#    sys.exit()

'''
###############################################################################
##################          SAVE FOR ENSEMBLE MODEL     ######################
###############################################################################
'''
if ENSEMBLE_AND_HYPEROPT_PREPARATION:
    h5f = h5py.File('generated_for_ensemble/lstm_dataset_prepared_for_ensemble.h5', 'w')
    h5f.create_dataset('train_x', data=train_x)
    h5f.create_dataset('valid_x', data=valid_x)
    h5f.create_dataset('test_x', data=test_x)
    h5f.create_dataset('train_y', data=train_y)
    h5f.create_dataset('valid_y', data=valid_y)
    h5f.create_dataset('test_y', data=test_y)
    h5f.close()
    
    print('Dataset is saved for hyperopt model')
    raise SystemExit



'''
###############################################################################
##################          TRAINING MODEL           ##########################
###############################################################################
'''
print('Training has started..')
early_stopping = EarlyStopping(monitor='val_loss', patience=PATIENCE, verbose=1)
checkpointer = ModelCheckpoint(filepath='generated_by_lstm_itemized/best_weights.hdf5', verbose=0, save_best_only=True)
model = Sequential()


if RUNNING_MODE == 'LSTM_RUN':
    model.add(LSTM(256, return_sequences=True, input_shape=(train_x.shape[-2], train_x.shape[-1])))
    model.add(Dropout(0.4))
    model.add(LSTM(64, return_sequences=True))
    model.add(Dropout(0.4))
    model.add(LSTM(128))
    model.add(Dropout(0.4))




elif RUNNING_MODE == '1D_CONVNET':
    model.add(Convolution1D(
                            input_shape=(train_x.shape[-2],train_x.shape[-1]),
                            nb_filter=8,
                            filter_length=2,    #filter
                            subsample_length=1, #stride
                            init='glorot_normal',
                            activation='relu')) 
    model.add(Convolution1D(
                            nb_filter=64,
                            filter_length=2,    
                            subsample_length=1, 
                            init='glorot_normal',
                            activation='relu')) 
    model.add(Convolution1D(
                            nb_filter=64,
                            filter_length=1,    
                            subsample_length=1, 
                            init='glorot_normal',
                            activation='relu')) 
    
    
    model.add(Flatten())


#Common
model.add(Dense(1, activation='sigmoid'))    
print(model.summary())
model.compile(loss='mae', optimizer='adam')
starttime = time.time()
history = model.fit(train_x,
                    train_y,
                    nb_epoch=EPOCHNO,
                    batch_size=BATCHSIZE,
                    callbacks=[early_stopping, checkpointer],
                    validation_data=(valid_x, valid_y),
                    verbose=1,
                    shuffle = True)
                    
model.load_weights('generated_by_lstm_itemized/best_weights.hdf5')
duration = time.time() - starttime
print('Time of running in minutes: ', duration/60)





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
ax.set_title(u'Hiba alakulása tanítás közben', y=1.05)
plt.tight_layout()
fig = ax.get_figure()
fig.savefig('generated_by_lstm_itemized/'+RUNNING_MODE+'_loss_curve.png')

#RETURN attr loss (STANDARDIZED form)    
result = model.predict(test_x, verbose = 0)
score = model.evaluate(test_x, test_y, verbose=0)
print('\n' + 'Test loss: ' + str(score))


#RETURN attr loss (SCALED BACK form)
df_return, figure = mu.scaleback_and_compare_target_prediction(target_scaler, result, test_y)
figure.show()
mse, corr_ret, result_describe, target_describe = mu.compare_target_prediction(df_return['Prediction_unscaled'].tolist(),df_return['Target_unscaled'].tolist())

mse_of_originalscale_return = np.mean( (df_return['Prediction_unscaled'] - df_return['Target_unscaled']) ** 2) 
print('Test RETURN MSE (original scale): ', mse_of_originalscale_return)

'''
###############################################################################
##################          CALCULATE  QUANTITY           #####################
###############################################################################
'''
#return_i = log(qty_i+1/qty_i)
#exp(return_i) = qty_i+1/qty_i
#qty_i+1 = exp(return_i) * qty_i

#Delabelize ITEM_IDs
ORIGINAL_PROD_GRP['test'] = label_encoder.inverse_transform(ORIGINAL_PROD_GRP['test'])
ORIGINAL_PROD_GRP['test'] = ORIGINAL_PROD_GRP['test'].astype('str')
GROUP_MAXES.index = label_encoder.inverse_transform(GROUP_MAXES.index)




#Predicted QTY SOFT METHOD
df_qty_grouped = pd.DataFrame({
                               'DATE'          :  ORIGINAL_DATE['test'],
                               'ITEM_ID'       :  ORIGINAL_PROD_GRP['test'],
                               'QTY_ORIGINAL'  :  ORIGINAL_QTY['test'],
                               'RET_PREDICTED' :  df_return['Prediction_unscaled'].tolist(),
                               'RET_ORIGINAL'  :  df_return['Target_unscaled'].tolist()
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


results = {}
results['CORR_RET'] = corr_ret[0]
results['CORR'] = pearsonr(df_qty_grouped['QTY_ORIGINAL'], df_qty_grouped['QTY_PREDICTED'])[0]
results['MAE'] = np.mean(abs(df_qty_grouped['QTY_ORIGINAL'] - df_qty_grouped['QTY_PREDICTED']))
results['RMSE'] = np.sqrt(np.mean((df_qty_grouped['QTY_ORIGINAL'] - df_qty_grouped['QTY_PREDICTED']) ** 2)) #mean not good for RMSE
results['MAPE'] = np.mean(abs((df_qty_grouped['QTY_ORIGINAL'] - df_qty_grouped['QTY_PREDICTED']) / df_qty_grouped['QTY_ORIGINAL'])) * 100

#Calculating MASE
n = ORIGINAL_QTY['train'].shape[0]
d = abs(ORIGINAL_QTY['train'].diff()).sum() / (n-1)
errors = np.abs(df_qty_grouped['QTY_ORIGINAL'] - df_qty_grouped['QTY_PREDICTED'])
results['MASE'] = errors.mean() / d


#Display curves
df_qty_grouped[['QTY_ORIGINAL', 'QTY_PREDICTED']].iloc[500:800].plot()



#Display results
print('CORR_RET', results['CORR_RET'])
print('CORR_QTY', results['CORR'])
print('MAE_QTY', results['MAE'])
print('RMSE_QTY', results['RMSE'])
print('MAPE: ', results['MAPE'])
print('MASE: ', results['MASE'])


'''
###############################################################################
##################          SAVE RESULTS                      #################
###############################################################################
'''
unique_folder = 'losses_{}_{:06.4f}_{:06.4f}_{}_{}_{}'.format(RUNNING_MODE, results['MASE'], mse_of_originalscale_return, int(results['MAE']), int(results['RMSE']), int(results['MAPE']))
path_folder = os.path.join('generated_by_lstm_itemized', unique_folder)
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

