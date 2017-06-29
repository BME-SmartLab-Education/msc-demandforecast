# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from pandas import read_hdf
import os
import matplotlib.pyplot as plt
import h5py
import pickle
SIZE = 20
plt.rc('font', size=20)          # controls default text sizes
plt.rc('axes', titlesize=30)     # fontsize of the axes title
plt.rc('axes', labelsize=30)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=20)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=25)    # legend fontsize
plt.rc('figure', titlesize=SIZE)    
plt.rcParams['figure.figsize'] = 16, 12
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.markersize'] = 10
plt.style.use('ggplot')


USE_ALREADY_SUMMARIZED_RESULTS = True
DRAW_MODEL_BY_MODEL_FORECAST = False
DRAW_PIECHART = False
DRAW_TOP5 = False
FRIEDMAN = True




'''
##########################################################################################################
##########################               GENERAL                                    ######################
##########################################################################################################
'''

models = [
            'Eltolás modell',
            'ARIMA',
            'Lineáris regresszió (egyben)',
            'Ridge regresszió (egyben)',
            'LASSO regresszió (egyben)',
            'Döntési fa (egyben)',
            'Véletlen erdő (egyben)',
            'XGBOOST (egyben)',
            'Lineáris regresszió (egyenként)',
            'Ridge regresszió (egyenként)',
            'LASSO regresszió (egyenként)',
            'Döntési fa (egyenként)',
            'Véletlen erdő (egyenként)',
            'XGBOOST (egyenként)',
            'FC',
            'LSTM',
            '1D CNN',
            'Aggregált modell'
        ]


#Load original train y
h5f = h5py.File('results_final/ORIGINAL_TRAIN_Y.h5','r')
ORIGINAL_TRAIN_Y = h5f['ORIGINAL_TRAIN_Y'][:]
h5f.close()
ORIGINAL_TRAIN_Y = pd.Series(ORIGINAL_TRAIN_Y)      
        
        
def calc_mae(group):
    return np.mean(abs(group['QTY_ORIGINAL'] - group['QTY_PREDICTED']))

def calc_rmse(group):
    return np.sqrt(np.mean((group['QTY_ORIGINAL'] - group['QTY_PREDICTED'])**2))

def calc_mape(group):
    return np.mean(abs((group['QTY_ORIGINAL'] - group['QTY_PREDICTED']) / group['QTY_ORIGINAL'])) * 100

def calc_mase(group):
    n = ORIGINAL_TRAIN_Y.shape[0]
    d = abs(ORIGINAL_TRAIN_Y.diff()).sum() / (n-1)
    errors = np.abs(group['QTY_ORIGINAL'] - group['QTY_PREDICTED'])
    return errors.mean() / d
    
def arima_calc_mase(prediction, loss):
    n = ORIGINAL_TRAIN_Y.shape[0]
    d = abs(ORIGINAL_TRAIN_Y.diff()).sum() / (n-1)
    errors = np.abs(prediction['QTY_ORIGINAL'] - prediction['QTY_PREDICTED'])
    return errors.mean() / d



    
'''
##########################################################################################################
##########################               FRESH EVALUATION                           ######################
##########################################################################################################
'''
if not USE_ALREADY_SUMMARIZED_RESULTS:

    grouped_losses = {}
    ls_losses = []
    dct_predictions = {}
    test = {}
    
    
    for model in models:
    
        #Read in
        path_loss = os.path.join('results_final', model, 'losses.h5')
        path_prediction = os.path.join('results_final', model, 'predictions.h5')
        df_loss = read_hdf(path_loss, 'losses')
        df_prediction = read_hdf(path_prediction, 'predictions')
        
        #hot fix for ARIMA MASE
        if model == 'arima':
            df_loss['MASE'] = arima_calc_mase(df_prediction, df_loss)
            
        
        #Collect losses
        df_loss.index=[model]
        ls_losses.append(df_loss)
        
        #Collect predictions
        dct_predictions[model] = df_prediction['QTY_PREDICTED'].tolist()
    
    
    
    #Create losses Dataframe
    df_all_losses = pd.concat(ls_losses)
    df_all_losses.drop(['CORR_RET'], axis=1,inplace=True)

    #Create predictions Dataframe
    df_all_predictions = pd.DataFrame(dct_predictions)
    df_all_predictions['DATE'] = df_prediction['DATE'].tolist()
    df_all_predictions['QTY_ORIGINAL'] = df_prediction['QTY_ORIGINAL'].tolist()
    df_all_predictions['ITEM_ID'] = df_prediction['ITEM_ID'].tolist()
    
        
    
    raise SystemExit    
    
    
    '''        
    ##########################################################################################################
    ##########################               USE SAVED EVALUATIION DATA                 ######################
    ##########################################################################################################
    '''
elif USE_ALREADY_SUMMARIZED_RESULTS:
    with open('results_final/predictions_evaluated.pickle', 'rb') as f:
        df_all_predictions = pickle.load(f)
    with open('results_final/losses_evaluated.pickle', 'rb') as f:
        df_all_losses = pickle.load(f)
    print('Previously created evaluation dataset is loaded from disk')

 

'''        
##########################################################################################################
##########################               VISUALIZATION                              ######################
##########################################################################################################
'''   
##########################################################################################################
##########################            SELECT TOP 5 MODEL                            ######################
##########################################################################################################
#Sort model names by losses (lowest first)
df_all_losses = df_all_losses.sort_values(by=['MAE','MASE', 'MAPE', 'RMSE'], ascending=True)

#Select top5
ls_top_names = df_all_losses.index[:5].tolist()

#XXX: choose interval
START_TIMESERIES_PLOT = 1000

##########################################################################################################
##########################            SIMPLE LINE CURVES - TOP5                     ######################
##########################################################################################################
if DRAW_TOP5:
    ls_columns = ls_top_names + ['QTY_ORIGINAL', 'DATE']
    df_top5 = df_all_predictions[ls_columns]
    
    #Select non zeros
    df_top5 = df_top5[df_top5['QTY_ORIGINAL'] != 0.1][START_TIMESERIES_PLOT:START_TIMESERIES_PLOT+50]
    
    #Draw image
    plt.rc('axes', labelsize=40)
    plt.rc('axes', titlesize=40) 
    plt.rc('xtick', labelsize=17) 
    plt.rc('legend', fontsize=30)
    ax = df_top5[['QTY_ORIGINAL', 'DATE']].plot(kind='bar', x='DATE', color='lightblue', width=1, use_index=False, alpha=0.75)
    df_top5[ls_top_names].plot(ax=ax, use_index=False)
    ls_legend = ls_top_names + [u'Eredeti rendelési mennyiség']
    ax.legend(ls_legend, loc="upper left")
    ax.set_title(u'Legeredményesebb 5 modell', y=1.03)
    ax.set_xlabel(u'Megfigyelés')
    ax.set_ylabel(u'Rendelési mennyiség')
    
    
    

##########################################################################################################
##########################               MODEL BY MODEL VIZ                         ######################
##########################################################################################################
if DRAW_MODEL_BY_MODEL_FORECAST:
#    for model in ls_top_names:
    for model in models:
        #Select non zeros; Columns -> [QTY_ORIGINAL, DATE, MODEL NAME]
        df_actual = df_all_predictions[df_all_predictions['QTY_ORIGINAL'] != 0.1][['QTY_ORIGINAL', 'DATE', model]]
        df_actual = df_actual[START_TIMESERIES_PLOT:START_TIMESERIES_PLOT+50]
        df_actual.rename(columns={model : 'QTY_PREDICTED'}, inplace=True)
        df_actual['DIFF'] = - abs(df_actual['QTY_PREDICTED'] - df_actual['QTY_ORIGINAL'])
        
        #Draw the predictions and comparison
        plt.rc('axes', titlesize=40) 
        plt.rc('xtick', labelsize=15) 
        plt.rc('axes', labelsize=40) 
        plt.rc('legend', fontsize=30)
        ax = df_actual[['QTY_PREDICTED','QTY_ORIGINAL','DATE']].plot(kind='bar', x='DATE', color=['skyblue', 'salmon'], use_index=False)
        df_actual[['DIFF','DATE']].plot(ax=ax, x='DATE', color='lightsteelblue', linestyle='-', marker='o', use_index=False)
        plt.title(model + u' előrejelzés', y=1.03)
        plt.legend([u'Abszolút hiba negatív előjellel',u'Előrejelzett rendelési mennyiség',u'Eredeti rendelési mennyiség'], loc="upper left")
        plt.xlabel(u'Megfigyelés')
        plt.ylabel(u'Rendelési mennyiség')
        plt.show()
        fig = ax.get_figure()
        fig.savefig('results_final/generated_images/'+model+'_forecast.png')
    
    


##########################################################################################################
##########################               PIE CHART VIZ                              ######################
##########################################################################################################
if DRAW_PIECHART:
    #XXX:
#    PIE_CHART_LOSS = calc_mae
#    PIE_CHART_LOSS = calc_rmse
#    PIE_CHART_LOSS = calc_mape
    PIE_CHART_LOSS = calc_mase

    
    #Map for loss names
    map_lossfnc_name = {
                         'calc_mae'  : 'MAE',   
                         'calc_rmse' : 'RMSE',   
                         'calc_mape' : 'MAPE',   
                         'calc_mase' : 'MASE',   
                        }
    
    dct_losses_byitems = {}
    #loop through top 5 models
    for model in ls_top_names:
        #Grouped losses
        df_actual = df_all_predictions[['ITEM_ID', 'QTY_ORIGINAL', model]]
        df_actual.rename(columns={model : 'QTY_PREDICTED'}, inplace=True)
        
        #Choose loss metric
        loss_any = df_actual.groupby('ITEM_ID').apply(PIE_CHART_LOSS)
        dct_losses_byitems[model] =  loss_any
            
    df_losses_byitems = pd.DataFrame(dct_losses_byitems)
    
    #Count min errors
    mins_per_row = np.argmin(df_losses_byitems.as_matrix(), axis=1)
    arr_counter = np.unique(mins_per_row, return_counts=True)
    dct_best_counter = dict(zip(df_losses_byitems.columns, arr_counter[1].tolist()))
    
    #Dataframe to Series
    df_best_perrow = pd.DataFrame(dct_best_counter, index=['COUNTER'])
    df_best_perrow = df_best_perrow.transpose()
    se_best_perrow = df_best_perrow['COUNTER']
    
    
    #Plot pie chart
    plt.rc('font', size=35)
    plt.figure(figsize=plt.figaspect(1))
    def make_autopct(values):
        def my_autopct(pct):
            total = sum(values)
            val = int(round(pct*total/100.0))
            return '{p:.2f}%  ({v:d})'.format(p=pct,v=val)
        return my_autopct
    
    patches, texts, autotexts = plt.pie(se_best_perrow.values, labels=se_best_perrow.index.tolist(), autopct=make_autopct(se_best_perrow.values), counterclock=False, colors=['powderblue','lightgreen','coral','peachpuff','thistle'])
    for text in texts:
        text.set_fontsize(40)
    plt.title(map_lossfnc_name[PIE_CHART_LOSS.func_name], y=1.03)
    plt.show()



'''
##########################################################################################################
##########################               FRIEDMAN                                   ######################
##########################################################################################################
'''
map_fullname_briefname = {
                        u'Eltolás modell'                        : 'eltolas_modell',
                        u'ARIMA'                                 : 'arima',
                        u'Lineáris regresszió (egyben)'          : 'linearis_regresszio_egyben',
                        u'Ridge regresszió (egyben)'             : 'ridge_regresszio_egyben',
                        u'LASSO regresszió (egyben)'             : 'lasso_regresszio_egyben',
                        u'Döntési fa (egyben)'                   : 'dontesi_fa_egyben',
                        u'Véletlen erdő (egyben)'                : 'veletlen_erdo_egyben',
                        u'XGBOOST (egyben)'                      : 'xgboost_egyben',
                        u'Lineáris regresszió (egyenként)'       : 'linearis_regresszio_egyenkent',
                        u'Ridge regresszió (egyenként)'          : 'ridge_regresszio_egyenkent',
                        u'LASSO regresszió (egyenként)'          : 'lasso_regresszio_egyenkent',
                        u'Döntési fa (egyenként)'                : 'dontesi_fa_egyenkent',
                        u'Véletlen erdő (egyenként)'             : 'veletlen_erdo_egyenkent',
                        u'XGBOOST (egyenként)'                   : 'xgboost_egyenkent',
                        u'FC'                                    : 'fc',
                        u'LSTM'                                  : 'lstm',
                        u'1D CNN'                                : 'cnn',
                        u'Aggregált modell'                      : 'aggregalt_modell'
                        }
if FRIEDMAN:
    #Calculate the DIFF
    map_all_diff_signed = {}
    map_all_diff_abs = {}
    map_all_diff_sqr = {}
    for model in models:
        df_actual = df_all_predictions[['QTY_ORIGINAL', model]]
        df_actual.rename(columns={model : 'QTY_PREDICTED'}, inplace=True)

        # with sign
        map_all_diff_signed[map_fullname_briefname[model]] = (df_actual['QTY_ORIGINAL'] - df_actual['QTY_PREDICTED']).values
        # with absolute value
        map_all_diff_abs[map_fullname_briefname[model]] = abs(df_actual['QTY_ORIGINAL'] - df_actual['QTY_PREDICTED']).values
        # with squared value
        map_all_diff_sqr[map_fullname_briefname[model]] = (df_actual['QTY_ORIGINAL'] - df_actual['QTY_PREDICTED']).values ** 2

    #Transform to DataFrame    
    df_all_diff_signed = pd.DataFrame(map_all_diff_signed)
    df_all_diff_abs = pd.DataFrame(map_all_diff_abs)
    df_all_diff_sqr = pd.DataFrame(map_all_diff_sqr)
    
    
    #Save to disk
    df_all_diff_signed.to_csv('spss_friedman_signed.csv', sep='\t', encoding='utf-8')
    df_all_diff_abs.to_csv('spss_friedman_abs.csv', sep='\t', encoding='utf-8')
    df_all_diff_sqr.to_csv('spss_friedman_sqr.csv', sep='\t', encoding='utf-8')
    print('Diffs for Friedman test are created')