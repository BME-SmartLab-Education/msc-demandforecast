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
from datetime import datetime
import os
import time





'''
###############################################################################
##################          SWITCHERS                 #########################
###############################################################################
'''
FORECAST_SHIFT = 7

#MODEL_TYPE = 'DECISION_TREE'
MODEL_TYPE = 'RANDOM_FOREST'
#MODEL_TYPE = 'XGBOOST'
#MODEL_TYPE = 'LASSO'
#MODEL_TYPE = 'RIDGE'
#MODEL_TYPE = 'LINEAR_REGRESSION'


ROLLIN_WINDOWS = 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15


'''
###############################################################################
##################          LOAD DATA                 #########################
###############################################################################
'''
with open('generated_data/data_prepared_itemized.pickle', 'rb') as f:
    dataset = pickle.load(f)
print('Dataset is loaded from disk')



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
dataset['ITEM_ID'] = label_encoder.fit_transform(dataset['ITEM_ID']) #Labelize



'''
###############################################################################
##################          INPUT FEATURES            #########################
###############################################################################
'''
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
print('Features are set')



'''
###############################################################################
##################          SPLIT DATA                #########################
###############################################################################
'''

#Split to train, valid, tst
train_set = dataset.groupby('ITEM_ID').apply(lambda x: x[:-40])
valid_set = dataset.groupby('ITEM_ID').apply(lambda x: x[-40:-20])
test_set = dataset.groupby('ITEM_ID').apply(lambda x: x[-20:])
train_set.reset_index(inplace=True, drop=True)
valid_set.reset_index(inplace=True, drop=True)
test_set.reset_index(inplace=True, drop=True)

#For MASE calculation    
ORIGINAL_TRAIN_Y = train_set['REQUESTED_QUANTITY'].copy()
print('Dataset is split')

#Number of item types
num_of_items = len(dataset['ITEM_ID'].unique())
del dataset


predictions = []
losses = []
start = time.time()
for i in range(num_of_items):
    '''
    ###############################################################################
    ##################          SELECT PRODUCT            #########################
    ###############################################################################
    '''
    print('\n{}. product is being fit'.format(i))    
    
    prod_name = i
    
    train_set_actual = train_set[train_set['ITEM_ID'] == prod_name].copy()
    valid_set_actual = valid_set[valid_set['ITEM_ID'] == prod_name].copy()
    test_set_actual = test_set[test_set['ITEM_ID'] == prod_name].copy()
    
    #Drop PROD_GROUP col
    input_names = [name for name in input_names if name != 'ITEM_ID']
    
    train_x = train_set_actual[input_names]
    train_y = train_set_actual[output_names].as_matrix().flatten()
    valid_x = valid_set_actual[input_names]
    valid_y = valid_set_actual[output_names].as_matrix().flatten()
    test_x = test_set_actual[input_names]
    test_y = test_set_actual[output_names].as_matrix().flatten()

    print('Selection and splitting is done')
    
    
    
    
    '''
    ###############################################################################
    ##################          SAVE ORIGINALS            #########################
    ###############################################################################
    '''
    #Save PROD_GRP, REQUESTED_QUANTITY, DATE
    MAX_QUANTITY_VALUE = train_set_actual['REQUESTED_QUANTITY'].max()
    ORIGINAL_QTY = test_set_actual['REQUESTED_QUANTITY'].copy()
    ORIGINAL_DATE = test_set_actual['REQUESTED_DELIVERY_DATE'].copy()
    print('Originals are saved')
    
    
    
    
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
    ##################          FITTING MODEL             #########################
    ###############################################################################
    '''
    print('Fitting model..')
        
    if MODEL_TYPE == 'LINEAR_REGRESSION':        
        model = LinearRegression()
        model.fit(train_x, train_y)
        predicted= model.predict(test_x)
        
    elif MODEL_TYPE == 'LASSO':
        model=Lasso(normalize=False, alpha=30)
        model.fit(train_x, train_y)
        predicted= model.predict(test_x)

      
    elif MODEL_TYPE == 'RIDGE':
        model=Ridge(fit_intercept=False, normalize=False, alpha=75)
        model.fit(train_x, train_y)
        predicted= model.predict(test_x)
        
    elif MODEL_TYPE == 'DECISION_TREE':    
        model = DecisionTreeRegressor(
                                        max_depth          = 29,
                                        min_impurity_split = 0.042,
                                        min_samples_split  = 0.2,
                                        min_samples_leaf   = 0.1
                                     )  
        model.fit(train_x, train_y)
        predicted= model.predict(test_x)
        
    elif MODEL_TYPE == 'RANDOM_FOREST':    
        model = RandomForestRegressor(
                                        n_estimators      = 100,
                                        verbose           = True,
                                        oob_score         = True,
                                        n_jobs            = -1,
                                        
                                        max_depth          = 29,
                                        min_impurity_split = 0.002,
                                        min_samples_split  = 0.2,
                                        min_samples_leaf   = 0.1     
                                     )
        model.fit(train_x, train_y)
        predicted= model.predict(test_x)
    
    
    
    elif MODEL_TYPE == 'XGBOOST':    
    
        xg_train = xgb.DMatrix(train_x, label=train_y)
        xg_valid = xgb.DMatrix(valid_x, label=valid_y)
        xg_test = xgb.DMatrix(test_x,   label=test_y)
        
        #Parameters
        param = {}
        
        param['objective']   = 'reg:linear'
        param['silent']      = 0
        param['nthread']     = -1
        
        param['eta']               = 0.3
        param['max_depth']         = 20
        param['colsample_bytree']  = 0.2
        param['min_child_weight']  = 15
        param['gamma']             = 462
        num_round = 1000
        watchlist = [(xg_train,'train'), (xg_valid, 'valid')]
        model = xgb.train(param, xg_train, num_round, watchlist, early_stopping_rounds=20)    
        predicted = model.predict(xg_test, ntree_limit=model.best_ntree_limit)
    
    
    '''
    ###############################################################################
    ##################          POSTPROCESSING (TRESHOLDING)    ###################
    ###############################################################################
    '''
    #Delabelize
    prod_name  = label_encoder.inverse_transform(prod_name)
    
    #Create result dataframe
    df_result = pd.DataFrame({
                                    'ITEM_ID'       : prod_name,
                                    'DATE'          : ORIGINAL_DATE,
                                    'QTY_ORIGINAL'  : ORIGINAL_QTY,
                                    'QTY_PREDICTED' : predicted.flatten()   #next quantity
                                })
    
    
    #Shift first (7th) record of every group, because NN can not predict this as, RETURN value is used instead of quantity
    df_result['QTY_PREDICTED'] = df_result['QTY_PREDICTED'].shift(FORECAST_SHIFT)
    df_result.dropna(how='any', inplace = True)

    
    
    #Clipping
    df_result['QTY_PREDICTED'] = df_result['QTY_PREDICTED'].clip(0, MAX_QUANTITY_VALUE)
    
    
    
         
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
    loss['ITEM_ID'] = prod_name
    print('CORR: ', loss['CORR'])
    print('MAE: ', loss['MAE'])
    print('RMSE: ', loss['RMSE'])
    print('MAPE: ', loss['MAPE'])


    predictions.append(df_result)  
    losses.append(loss)
    

    
    

df_predictions = pd.concat(predictions)
df_losses = pd.concat([pd.DataFrame(x, index=[0]) for x in losses])
df_losses.set_index('ITEM_ID', inplace=True)

final_loss = {}
final_loss['CORR'] = pearsonr(df_predictions['QTY_ORIGINAL'], df_predictions['QTY_PREDICTED'])[0]
final_loss['MAE'] =  np.mean(abs(df_predictions['QTY_ORIGINAL'] - df_predictions['QTY_PREDICTED']))
final_loss['RMSE'] =  np.sqrt(np.mean((df_predictions['QTY_ORIGINAL'] - df_predictions['QTY_PREDICTED']) ** 2))
final_loss['MAPE'] = np.mean(abs((df_predictions['QTY_ORIGINAL'] - df_predictions['QTY_PREDICTED']) / df_predictions['QTY_ORIGINAL'])) * 100

n = ORIGINAL_TRAIN_Y.shape[0]
d = abs(ORIGINAL_TRAIN_Y.diff()).sum() / (n-1)
errors = np.abs(df_predictions['QTY_ORIGINAL'] - df_predictions['QTY_PREDICTED'])
final_loss['MASE'] = errors.mean() / d

print('\nfinal CORR: ', final_loss['CORR'])
print('final MAE: ',  final_loss['MAE'])
print('final RMSE: ', final_loss['RMSE'])
print('final MAPE: ', final_loss['MAPE'])
print('final MASE: ', final_loss['MASE'])

duration = (time.time() - start) / 60
print('Running took {} minutes'.format(duration))

'''
###############################################################################
##################          SAVE RESULTS                      #################
###############################################################################
'''
unique_folder = 'losses_{}_{:06.4f}_{}_{}_{}'.format(MODEL_TYPE, final_loss['MASE'], int(final_loss['MAE']), int(final_loss['RMSE']), int(final_loss['MAPE']))
path_folder = os.path.join('generated_by_baselines_1by1_itemized', unique_folder)
os.mkdir(path_folder)

path_losses = os.path.join(path_folder, 'losses.h5')
path_predictions = os.path.join(path_folder, 'predictions.h5')

#Save losses and predictions
final_loss = pd.DataFrame(final_loss,index=[0])
final_loss.to_hdf(path_losses, 'losses', mode='w')
df_predictions.to_hdf(path_predictions, 'predictions', mode='w')


#Save source code
path_source_code = os.path.join(path_folder, 'source_code.txt')
with open(__file__) as sc:
    source_code = sc.read()
with open(path_source_code, 'w') as text_file:
    text_file.write(source_code)

print('Results are saved to disk')
