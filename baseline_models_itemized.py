# -*- coding: utf-8 -*-


from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor 
import xgboost as xgb
import pickle
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import matemodules.mateutils_byitems as mu
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn import preprocessing
import time
from datetime import datetime
import os
import h5py



'''
###############################################################################
##################          SWITCHERS                 #########################
###############################################################################
'''
FORECAST_SHIFT = 7
TEST_SET_SIZE = 20

#MODEL_TYPE = 'DECISION_TREE'
#MODEL_TYPE = 'RANDOM_FOREST'
#MODEL_TYPE = 'XGBOOST'
#MODEL_TYPE = 'LASSO'
MODEL_TYPE = 'RIDGE'
#MODEL_TYPE = 'LINEAR_REGRESSION'

HYPEROPT_PREPROCESS = False


#NUMBER_OF_RETS = 20
ROLLIN_WINDOWS = 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15



'''
###############################################################################
##################          LOAD DATA                 #########################
###############################################################################
'''
with open('generated_data/data_prepared_itemized.pickle', 'rb') as f:
    dataset = pickle.load(f)
print('Dataset is loaded from disk')

with open('generated_by_fc_itemized/embedding_weights_itemized.pickle', 'rb') as f:
    embeddings = pickle.load(f)
print('Embedding data is loaded from disk')



'''
###############################################################################
##################          ADD COLUMNS               #########################
###############################################################################
'''
mu.add_rollin_mean_cols(dataset, 'ITEM_ID', 'REQUESTED_QUANTITY', *ROLLIN_WINDOWS)

#Drop NaNs
dataset.dropna(inplace=True)
dataset.reset_index(inplace=True, drop=True)
print('New columns are added')


'''
###############################################################################
##################          LABELIZED ITEMIDS         #########################
###############################################################################
'''
#Encode ITEM_IDs from string to one hot
label_encoder = LabelEncoder()
dataset['ITEM_ID'] = label_encoder.fit_transform(dataset['ITEM_ID']) 
dataset['YEAR'] = LabelEncoder().fit_transform(dataset['YEAR']) 
dataset['MONTH'] = LabelEncoder().fit_transform(dataset['MONTH']) 
dataset['DAY'] = LabelEncoder().fit_transform(dataset['DAY']) 
dataset['DAY_OF_WEEK'] = LabelEncoder().fit_transform(dataset['DAY_OF_WEEK']) 
dataset['IS_WEEKEND'] = LabelEncoder().fit_transform(dataset['IS_WEEKEND']) 
print('Features are labelized')




'''
###############################################################################
##################          SPLIT DATA                #########################
###############################################################################
'''

#Split to train, valid, tst
train_set = dataset.groupby('ITEM_ID').apply(lambda x: x[:-(2*TEST_SET_SIZE)])
valid_set = dataset.groupby('ITEM_ID').apply(lambda x: x[-(2*TEST_SET_SIZE):-TEST_SET_SIZE])
test_set = dataset.groupby('ITEM_ID').apply(lambda x: x[-TEST_SET_SIZE:])
train_set.reset_index(inplace=True, drop=True)
valid_set.reset_index(inplace=True, drop=True)
test_set.reset_index(inplace=True, drop=True)
print('Dataset is split')


'''
###############################################################################
##################          SAVE ORIGINALS            #########################
###############################################################################
'''
#Save PROD_GRP, REQUESTED_QUANTITY, DATE
ORIGINAL_PROD_GRP = test_set['ITEM_ID'].copy()
ORIGINAL_QTY = test_set['REQUESTED_QUANTITY'].copy()
ORIGINAL_DATE = test_set['REQUESTED_DELIVERY_DATE'].copy()

#Save goup maxes for thresholding (Postprocessing)
GROUP_MAXES = train_set.groupby('ITEM_ID')['REQUESTED_QUANTITY'].max()
GROUP_MAXES.index  = label_encoder.inverse_transform(GROUP_MAXES.index)

#Save for MASE loss metric
ORIGINAL_TRAIN_Y = train_set['REQUESTED_QUANTITY'].copy()

print('Originals are saved')


'''
###############################################################################
##################          FEATURE SELECTION         #########################
###############################################################################
'''
#Drop PROD_GROUP col
col_names = dataset.columns
movin_mean_cols = [name for name in col_names if name.startswith('ROLLING_MEAN') ]
input_names = [                
                       'ITEM_ID',
                       'YEAR',
                       'MONTH',
                       'DAY',
                       'DAY_OF_WEEK',
                       'IS_WEEKEND',
                       'NEXT_HOLIDAY',
                       'PREV_HOLIDAY'
                       ] + movin_mean_cols +  ['ZEROS_CUMSUM', 'ZERO_FULL_RATIO', 'MAX_ZERO_SEQUENCE', 'MEAN_OF_ZERO_SEQ', 'ZERO_QTY_SEQUENCE', 'ORDERS_THAT_DAY']
output_names = ['QTY_PRED'] 
print(input_names)
print('Features are set')



train_x = train_set[input_names]
valid_x = valid_set[input_names]
test_x = test_set[input_names]
train_y = train_set[output_names].as_matrix().flatten()
valid_y = valid_set[output_names].as_matrix().flatten()
test_y = test_set[output_names].as_matrix().flatten()


def embedding_input(data):
    index_embedding_mapping = {
                                data.columns.get_loc('ITEM_ID'):       0,
                                data.columns.get_loc('YEAR'):          1,
                                data.columns.get_loc('MONTH'):         2,
                                data.columns.get_loc('DAY'):           3,
                                data.columns.get_loc('DAY_OF_WEEK'):   4,
                                data.columns.get_loc('IS_WEEKEND'):    5
                                }
                                
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

train_x = embedding_input(train_x)
valid_x = embedding_input(valid_x)
test_x = embedding_input(test_x)
print('Features are selected')


'''
###############################################################################
##################          STANDARDIZATION           #########################
###############################################################################
'''
scaler_standard = preprocessing.StandardScaler().fit(train_x)
train_x = scaler_standard.transform(train_x)
valid_x = scaler_standard.transform(valid_x)
test_x = scaler_standard.transform(test_x)
print('Data is standardized')




'''
###############################################################################
##################          SAVE FOR HYPEROPT         #########################
###############################################################################
'''
if HYPEROPT_PREPROCESS:
    path_directory = 'generated_by_baselines_batch_itemized'
    path_dataset = path_directory + '/hyperopt_dataset_baselines.h5' 
    path_scaler = path_directory + '/hyperopt_scaler_baselines.pickle'
    
    h5f = h5py.File(path_dataset, 'w')
    h5f.create_dataset('train_x', data=train_x)
    h5f.create_dataset('valid_x', data=valid_x)
    h5f.create_dataset('test_x', data=test_x)
    h5f.create_dataset('train_y', data=train_y)
    h5f.create_dataset('valid_y', data=valid_y)
    h5f.create_dataset('test_y', data=test_y)
    h5f.close()
    print('Data set for HYPEROPT is generated and saved')
    
    
    with open(path_scaler, 'wb') as f:
        pickle.dump(scaler_standard, f, 2)
    print('Scaler object for HYPEROPT is saved')
    
    raise SystemExit


'''
###############################################################################
##################          FITTING MODEL             #########################
###############################################################################
'''
print('Fitting {} model'.format(MODEL_TYPE))
start = time.time()

###############################################################################
########      LINEAR REGRESSION       #########################################
###############################################################################
if MODEL_TYPE == 'LINEAR_REGRESSION':        
    model = LinearRegression()
    model.fit(train_x, train_y)
    predicted= model.predict(test_x)


###############################################################################
########      LASSO       #####################################################
###############################################################################
if MODEL_TYPE == 'LASSO':
    model=Lasso(normalize=False, alpha=0.1)
    model.fit(train_x, train_y)
    predicted= model.predict(test_x)


###############################################################################
########      RIDGE       #####################################################
###############################################################################
elif MODEL_TYPE == 'RIDGE':
    model=Ridge(fit_intercept=False, normalize=False, alpha=20.9)
    model.fit(train_x, train_y)
    predicted= model.predict(test_x)


###############################################################################
########      DECISION TREE        ############################################
###############################################################################
elif MODEL_TYPE == 'DECISION_TREE':    
    model = DecisionTreeRegressor(
                                    max_depth          = 272,
                                    min_impurity_split = 0.015,
                                    min_samples_split  = 59,
                                    max_features       = 10,
                                    min_samples_leaf   = 88,
                                    max_leaf_nodes     = 73300
                                )  
                                

    model.fit(train_x, train_y)
    predicted= model.predict(test_x)


###############################################################################
########      RANDOM FOREST            ########################################    
###############################################################################
elif MODEL_TYPE == 'RANDOM_FOREST':    
    model = RandomForestRegressor(
                                    n_estimators      = 1000,
                                    verbose           = True,
                                    oob_score         = True,
                                    n_jobs            = -1,
                                    
                                    max_depth          = 62,
                                    min_impurity_split = 0.074,
                                    min_samples_split  = 76,
                                    max_features       = 5,
                                    min_samples_leaf   = 73,
                                    max_leaf_nodes     = 97483
                                )
                                
    model.fit(train_x, train_y) 
    predicted = model.predict(test_x)
    

###############################################################################
########      XGBOOST      ####################################################
###############################################################################
elif MODEL_TYPE == 'XGBOOST':    
    xg_train = xgb.DMatrix(train_x, label=train_y)
    xg_valid = xgb.DMatrix(valid_x, label=valid_y)
    xg_test = xgb.DMatrix(test_x,   label=test_y)
    
    #Parameters
    param = {}
    
    param['objective']   = 'reg:linear'
    param['silent']      = 0
    param['nthread']     = -1
    
    
    param['eta']               = 0.35
    param['max_depth']         = 83
    param['colsample_bytree']  = 0.5
    param['subsample']         = 0.45
    param['min_child_weight']  = 347
    param['gamma']             = 620
    num_round = 100
    
    watchlist = [(xg_train,'train'), (xg_valid, 'valid')]
    model = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=20)    
    predicted = model.predict(xg_test, ntree_limit=model.best_ntree_limit)


duration = (time.time() - start) / 60
print('Running took {} minutes'.format(duration))


'''
###############################################################################
##################          POSTPROCESSING (TRESHOLDING)    ###################
###############################################################################
'''
#Labelize backward
ORIGINAL_PROD_GRP  = label_encoder.inverse_transform(ORIGINAL_PROD_GRP)
ORIGINAL_PROD_GRP = ORIGINAL_PROD_GRP.astype('str')

#Create result dataframe
df_result = pd.DataFrame({      
                                'ITEM_ID'       : ORIGINAL_PROD_GRP,
                                'DATE'          : ORIGINAL_DATE,
                                'QTY_ORIGINAL'  : ORIGINAL_QTY,          
                                'QTY_PREDICTED' : predicted
                            })


#Shift first (7th) record of every group, because NN can not predict this as, RETURN value is used instead of quantity
df_result['QTY_PREDICTED'] = df_result.groupby('ITEM_ID')['QTY_PREDICTED'].shift(FORECAST_SHIFT)
df_result.dropna(how='any', inplace = True)



#Group by group clipping
def threshold(group):
    max_value = GROUP_MAXES[group.name]
    return group.clip(0,max_value)   
df_result['QTY_PREDICTED'] = df_result.groupby('ITEM_ID')['QTY_PREDICTED'].apply(threshold)




    
'''
###############################################################################
##################          EVALUATE                  #########################
###############################################################################
'''
#Losses
loss = {}
loss['CORR'] = pearsonr(df_result['QTY_ORIGINAL'], df_result['QTY_PREDICTED'])[0]
loss['MAE'] =  np.mean(abs(df_result['QTY_ORIGINAL'] - df_result['QTY_PREDICTED']))
loss['RMSE'] =  np.sqrt(np.mean((df_result['QTY_ORIGINAL'] - df_result['QTY_PREDICTED']) ** 2))
loss['MAPE'] = np.mean(abs((df_result['QTY_ORIGINAL'] - df_result['QTY_PREDICTED']) / df_result['QTY_ORIGINAL'])) * 100
#Calculating MASE
n = ORIGINAL_TRAIN_Y.shape[0]
d = abs(ORIGINAL_TRAIN_Y.diff()).sum() / (n-1)
errors = np.abs(df_result['QTY_ORIGINAL'] - df_result['QTY_PREDICTED'])
loss['MASE'] = errors.mean() / d


print('CORR: ', loss['CORR'])
print('MAE: ', loss['MAE'])
print('RMSE: ', loss['RMSE'])
print('MAPE: ', loss['MAPE'])
print('MASE', loss['MASE'])


#Plotting
df_result.reset_index(inplace=True, drop=True)
df_result.iloc[300:600].plot()


