# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.arima_model import ARIMA
import os
import warnings
import time
from scipy.stats import pearsonr
import sys
import pickle
from pandas import HDFStore
from pandas import read_hdf
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")



FORECAST_AHEAD = 7
LEN_TEST = 20

'''
#############################################################################################
###########                     LOADING FROM DISK                               #############
#############################################################################################
'''
#Load data
with open('generated_data/data_prepared_itemized.pickle', 'rb') as f:
    orders = pickle.load(f)
print('Dataset is loaded from disk')


'''
#############################################################################################
###########                     ARIMA CLASS                                     #############
#############################################################################################
'''
class ARIMABaseLiner():
    def __init__(self):
        self.counter = 0
        self.errorcounter = 0        
        self.correctioncounter = 0        
        
    def counter_plusplus(self):
        self.counter += 1
    
    def get_counter_str(self):
        return str(self.counter)        

   
   
   #Searching the ideal parameters for the given timeseries based on aic value
    def search_parameters(self, prod_name):
      
        
        '''        
        #############################################################################################
        ###########                     GENERAL                                         #############
        #############################################################################################
        '''
        plt.ioff()
        self.counter_plusplus()
        print('Product #', self.get_counter_str())
        print('Processing: ', prod_name)
        starttime = time.time()
        
        
        
        
        '''
        #############################################################################################
        ###########                     SELECTING                                       #############
        #############################################################################################
        '''
        df_orders = orders[orders['ITEM_ID'] == prod_name].copy()
        df_orders.set_index('REQUESTED_DELIVERY_DATE', inplace=True)
        df_orders = df_orders[['REQUESTED_QUANTITY', 'QTY_LOG']] 
                
        
        
        '''
        #############################################################################################
        ###########                     SET TRAIN/TEST RATIO                            #############
        #############################################################################################
        '''
        LEN_FULL = df_orders.shape[0]
        LEN_TRAIN_VALID = LEN_FULL - LEN_TEST
        
        
        '''
        #############################################################################################
        ###########                     TEST PARAMS                                     #############
        #############################################################################################
        '''
        print('Searching for ideal parameters..')
        testcases = {}
        d=1
        for p in range(1,7):
                for q in range(1,5):
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        try:
                            test_model = ARIMA(df_orders['QTY_LOG'][:LEN_TRAIN_VALID], order=(p, d, q))  
                            test_results_ARIMA = test_model.fit(disp=-1)                              
                            aic = test_results_ARIMA.aic
                            param = p,d,q
                            testcases[param] = aic
                        except:
                            pass #ignore the error and go on
                            
                                         
        best_params = min(testcases, key=testcases.get)
        p,d,q = best_params
        print('Best p,d,q params are: ')
        print('(',p,d,q,')')
        print('Running took {} seconds'.format(time.time()-starttime))        
        print('-' * 60)
        
        return pd.DataFrame({'p':p,'d':d, 'q':q}, index=[prod_name])
        
        
        
        
        
        
    #Fitting ARIMA models using the already found ideal p,d,q parameters
    def fit_arima(self, prod_name, parameters):
        '''        
        #############################################################################################
        ###########                     GENERAL                                         #############
        #############################################################################################
        '''
        plt.ioff()
        self.counter_plusplus()
        print('Product #', self.get_counter_str())
        print('Processing: ', prod_name)
        starttime = time.time()
        
        
        
        
        '''
        #############################################################################################
        ###########                     SELECTING                                       #############
        #############################################################################################
        '''
        df_orders = orders[orders['ITEM_ID'] == prod_name].copy()
        df_orders.set_index('REQUESTED_DELIVERY_DATE', inplace=True)
        df_orders = df_orders[['REQUESTED_QUANTITY', 'QTY_LOG']] 
                
        
        
        '''
        #############################################################################################
        ###########                     SET TRAIN/TEST RATIO                            #############
        #############################################################################################
        '''
        LEN_FULL = df_orders.shape[0]
        LEN_TRAIN_VALID = LEN_FULL - LEN_TEST
        
        #For being consequent with NN approach
        ts_test = df_orders[LEN_TRAIN_VALID+FORECAST_AHEAD:]
        
        
        
        
        
        '''
        #############################################################################################
        ###########         ERROR HANDLING, FIND ANOTHER PARAMETER TUPLE                #############
        #############################################################################################
        '''        
        def find_other_pdq(p, q):
            if p > q:
                p-=1
            else:
                q-=1
            return p,q
            
            
        def forecast_x_ahead_one_obs(i,p,q):
            try:
                results_ARIMA = ARIMA(df_orders['QTY_LOG'][:LEN_TRAIN_VALID+1+i], order=(p, d, q)).fit(disp=-1)  
                res = results_ARIMA.forecast(FORECAST_AHEAD)[0] 
                return res[-1], p, q
            except (ValueError, np.linalg.LinAlgError) as e:
                self.errorcounter += 1
                print('Error occured ({})'.format(self.errorcounter))
                p,q = find_other_pdq(p,q)
                res, p, q = forecast_x_ahead_one_obs(i,p,q)
                return res, p, q
        
        '''
        #############################################################################################
        ###########                     ARIMA FIT AND FORECAST                          #############
        #############################################################################################
        '''
        
        forecasted_values = []
        for i in range(LEN_TEST-FORECAST_AHEAD): #9 or 13
            p,d,q = parameters
            p_original = p
            q_original = q
            forcasted_value, p, q = forecast_x_ahead_one_obs(i,p,q)
            if(p_original!=p or q_original!=q):
                    self.correctioncounter += 1
                    print('One correction happened ({})'.format(self.correctioncounter))
            forecasted_values.append(forcasted_value)
            

        ts_test['QTY_LOG_PRED'] = np.array(forecasted_values)
        ts_test[['QTY_LOG','QTY_LOG_PRED']].plot()
        
        
        
        
        '''
        #############################################################################################
        ###########                    EVALUATION        (TEST SET)                     #############
        #############################################################################################
        '''
        #Predicted calc back
        ts_test['QTY_PREDICTED'] = np.exp(ts_test['QTY_LOG_PRED'])
        
        #Fill the Nans generated because of errors
        mean_actual = df_orders['REQUESTED_QUANTITY'][:LEN_TRAIN_VALID].mean()
        ts_test['QTY_PREDICTED'].fillna(mean_actual, inplace=True)  
        
        #Thresholding
        max_actual = df_orders['REQUESTED_QUANTITY'][:LEN_TRAIN_VALID].max()
        ts_test['QTY_PREDICTED'] = ts_test['QTY_PREDICTED'].clip(0,max_actual)
        
        #Evaluate losses
        corr_qtylog = pearsonr(ts_test['QTY_LOG'], ts_test['QTY_LOG_PRED'])
        corr_qty = pearsonr(ts_test['QTY_PREDICTED'], ts_test['REQUESTED_QUANTITY'])
        mae_final = np.mean(abs(ts_test['QTY_PREDICTED']-ts_test['REQUESTED_QUANTITY']))
        rmse_final = np.sqrt(np.mean((ts_test['QTY_PREDICTED']-ts_test['REQUESTED_QUANTITY'])**2))
        mape_final = np.mean(abs((ts_test['REQUESTED_QUANTITY']-ts_test['QTY_PREDICTED']) / ts_test['REQUESTED_QUANTITY'])) * 100
        
        
        
        #Plot prediction
        ax = ts_test[['REQUESTED_QUANTITY','QTY_PREDICTED']].plot()
        plt.title('RMSE: {:.4f} - ({})'.format(rmse_final, prod_name))
        fig = ax.get_figure()
        file_name = os.path.join('arima', self.get_counter_str() + '.jpg')
        fig.savefig(file_name)
        plt.close(fig)        
        
        
        
        #Print results
        print('Correlation of Quantity Log: {}'.format(corr_qtylog))
        print('Correlation of Quantity: {}'.format(corr_qty))
        print('MAE Quantity: {:.4f}'.format(mae_final))
        print('RMSE Quantity: {:.4f}'.format(rmse_final))
        print('MAPE Quantity: {:.4f}'.format(mape_final))
        print('Running took {} seconds'.format(time.time()-starttime))        
        print('-' * 60)
        
         
            
        df_preds = pd.DataFrame({    'ITEM_ID'                  : prod_name,
                                     'QTY_LOG_ORIGINAL'         : ts_test['QTY_LOG'],
                                     'QTY_LOG_PRED'             : ts_test['QTY_LOG_PRED'],
                                     'QTY_ORIGINAL'             : ts_test['REQUESTED_QUANTITY'],
                                     'QTY_PREDICTED'            : ts_test['QTY_PREDICTED']
                                })


        df_losses = pd.DataFrame({
                                    'PRODUCT'        :  prod_name,
                                    'PDQ'            :  '{},{},{}'.format(p,d,q),
                                    'RMSE'           :  rmse_final,
                                    'MAE'            :  mae_final,
                                    'MAPE'           :  mape_final,
                                    'CORR_QTYLOG'    :  corr_qtylog[0],
                                    'CORR_QTY'       :  corr_qty[0]
                                },index=[0])
                
              
        return df_preds, df_losses
                        



        



'''
#############################################################################################
###########                    RUN ARIMA FOR ALL ITEMS                          #############
#############################################################################################
'''
#mode of running
RUNNING_MODE = 'PARAMETER_SEARCH'
#RUNNING_MODE = 'ARIMA_FITTING'

NB_OF_ITEMS = 2



#Run for all in a loop
prod_names = orders['ITEM_ID'].unique()
arima_model = ARIMABaseLiner()
start = time.time()


#Only search
if RUNNING_MODE == 'PARAMETER_SEARCH':
    parameters = []    
    for name in prod_names:
        param = arima_model.search_parameters(name)
        parameters.append(param)
        
        
    #Save PARAMETERS (p,d,q)
    path_params = os.path.join('generated_by_arima_itemized', 'arima_parameters.h5')
    df_parameters = pd.concat(parameters)
    df_parameters.to_hdf(path_params, 'params', mode='w')
        
        
      
#Only fitting   
elif RUNNING_MODE == 'ARIMA_FITTING':    
    #Load parameters
    path_params = os.path.join('generated_by_arima_itemized', 'arima_parameters.h5')
    df_parameters = read_hdf(path_params, 'params')

    
    predictions = []
    losses = []
#    for name in prod_names[650:]:
    for name in prod_names:
        #Check whether we have parameter for that product
        if name in df_parameters.index:
            param = df_parameters.loc[name][['p','d','q']].values
            prediction, loss = arima_model.fit_arima(name,param)
            predictions.append(prediction)
            losses.append(loss)
    
    
    '''
    ###############################################################################
    ##################          EVALUATION                        #################
    ###############################################################################
    '''
    #Final result set
    df_predictions = pd.concat(predictions)
    df_losses = pd.concat(losses)
    
    #Losses
    finals = {}
    finals['MAE'] = np.mean(abs(df_predictions['QTY_ORIGINAL']-df_predictions['QTY_PREDICTED']))
    finals['RMSE'] = np.sqrt(np.mean((df_predictions['QTY_ORIGINAL']-df_predictions['QTY_PREDICTED']) ** 2))
    finals['MAPE'] = np.mean(abs((df_predictions['QTY_ORIGINAL']-df_predictions['QTY_PREDICTED']) / df_predictions['QTY_ORIGINAL'])) * 100
    finals['CORR'] = pearsonr(df_predictions['QTY_ORIGINAL'], df_predictions['QTY_PREDICTED'])[0]
    df_finals = pd.DataFrame(finals, index=[0])
    
    #Print results
    print('MAE final: ', finals['MAE'])
    print('RMSE final: ', finals['RMSE'])
    print('MAPE final: ', finals['MAPE'])
    print('CORR final: ', finals['CORR'])
    
    
    
    
    '''
    ###############################################################################
    ##################          SAVE RESULTS                      #################
    ###############################################################################
    '''
    #Create folder with date and model type
    date_time = datetime.now()
    unique_time = date_time.strftime('%Y-%m-%d_%H-%M')
    date_time_folder = unique_time + '_ARIMA_' +  str(FORECAST_AHEAD) #like: 2017-02-27_13-ARIMA_1
    path_folder = os.path.join('generated_by_arima_itemized', date_time_folder)
    os.mkdir(path_folder)
    
    path_losses = os.path.join(path_folder, 'losses.h5')
    path_predictions = os.path.join(path_folder, 'predictions.h5')
    
    #Save losses and predictions
    df_finals.to_hdf(path_losses, 'losses', mode='w')
    df_predictions.to_hdf(path_predictions, 'predictions', mode='w')
    print('Results are saved to disk')
    
    
    
    
    
#Common
duration = (time.time()-start) /60
print('The test took {} minutes'.format(duration))    
print('{} errors occured'.format(arima_model.errorcounter))    
print('{} corrections happened'.format(arima_model.correctioncounter))    

