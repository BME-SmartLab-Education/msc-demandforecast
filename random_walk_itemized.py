# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
from scipy.stats import pearsonr
from datetime import datetime
import pickle


FORECAST_SHIFT = 7
SIZE_TEST_SET = 20



'''
#############################################################################################
###########                     LOADING FROM DISK                               #############
#############################################################################################
'''
#Load data
with open('generated_data/data_prepared_itemized.pickle', 'rb') as f:
    orders = pickle.load(f)
print('Dataset is loaded from disk')


#Create the test set
orders = orders.groupby('ITEM_ID').apply(lambda x: x[-SIZE_TEST_SET:])



'''
#############################################################################################
###########                     RANDOM WALK                               #############
#############################################################################################
'''

#Create new dataframe
df_result = pd.DataFrame({
                               'DATE'          :  orders['REQUESTED_DELIVERY_DATE'],
                               'ITEM_ID'       :  orders['ITEM_ID'],
                               'QTY_ORIGINAL'  :  orders['REQUESTED_QUANTITY'],
                               'QTY_PREDICTED' :  orders['REQUESTED_QUANTITY']
                               })

#Do the random walk shifting
df_result['QTY_PREDICTED'] = df_result.groupby('ITEM_ID')['QTY_PREDICTED'].shift(FORECAST_SHIFT)
df_result.dropna(how='any', inplace=True)



'''
#############################################################################################
###########                    EVALUATION                                       #############
#############################################################################################
'''
#SINGLE VALUES
loss = {}
loss['CORR'] = pearsonr(df_result['QTY_ORIGINAL'], df_result['QTY_PREDICTED'])[0]
loss['MAE'] =  np.mean(abs(df_result['QTY_ORIGINAL'] - df_result['QTY_PREDICTED']))
loss['RMSE'] =  np.sqrt(np.mean((df_result['QTY_ORIGINAL'] - df_result['QTY_PREDICTED']) ** 2))
loss['MAPE'] = np.mean(abs((df_result['QTY_ORIGINAL'] - df_result['QTY_PREDICTED']) / df_result['QTY_ORIGINAL'])) * 100

#Calc MASE
import h5py
h5f = h5py.File('results_final/ORIGINAL_TRAIN_Y.h5','r')
ORIGINAL_TRAIN_Y = h5f['ORIGINAL_TRAIN_Y'][:]
h5f.close()
ORIGINAL_TRAIN_Y = pd.Series(ORIGINAL_TRAIN_Y)
n = ORIGINAL_TRAIN_Y.shape[0]
d = abs(ORIGINAL_TRAIN_Y.diff()).sum() / (n-1)
errors = np.abs(df_result['QTY_ORIGINAL'] - df_result['QTY_PREDICTED'])
loss['MASE'] = errors.mean() / d


print('CORR: ', loss['CORR'])
print('MAE: ',  loss['MAE'])
print('RMSE: ', loss['RMSE'])
print('MAPE: ', loss['MAPE'])
print('MASE: ', loss['MASE'])


#Plotting
df_result.reset_index(inplace=True, drop=True)
df_result.iloc[600:700].plot()



'''
###############################################################################
##################          SAVE RESULTS                      #################
###############################################################################
'''
unique_folder = 'losses_{:06.4f}_{}_{}_{}'.format(loss['MASE'], int(loss['MAE']), int(loss['RMSE']), int(loss['MAPE']))
path_folder = os.path.join('generated_by_randomwalk_itemized', unique_folder)
os.mkdir(path_folder)

path_losses = os.path.join(path_folder, 'losses.h5')
path_predictions = os.path.join(path_folder, 'predictions.h5')

#Save losses and predictions
df_loss = pd.DataFrame(loss,index=[0])
df_loss.to_hdf(path_losses, 'losses', mode='w')
df_result.to_hdf(path_predictions, 'predictions', mode='w')

#Save source code
path_source_code = os.path.join(path_folder, 'source_code.txt')
with open(__file__) as sc:
    source_code = sc.read()
    
with open(path_source_code, 'w') as text_file:
    text_file.write(source_code)
print('Results are saved to disk')