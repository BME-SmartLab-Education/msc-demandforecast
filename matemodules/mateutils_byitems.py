# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from scipy.stats import describe, pearsonr




#This lambda function replaces NaNs to zeros
prevent_from_nans = lambda x: x if pd.notnull(x) else 0

    

def add_prev_ret_cols(number, data, grpby_colname, attr_name):
    for i in range(number):
        prev_close = i 
        prev_far = i+1
        new_col_name = 'RET-' + str(prev_close)
        data[new_col_name] = data.groupby([grpby_colname])[attr_name].apply(lambda x: np.log(x.shift(prev_close)) - np.log(x.shift(prev_far)))#.apply(prevent_from_nans)

               


def add_rollin_mean_cols(data, grpby_colname, attr_name, *windows):
    for window in windows:
        new_col_name = 'ROLLING_MEAN-' + str(window)
        data[new_col_name] = data.groupby([grpby_colname])[attr_name].apply(lambda x: x.rolling(center=False, window = window).mean())#.apply(prevent_from_nans)
        
        

def add_rollin_std_cols(data, grpby_colname, attr_name, *windows):
    for window in windows:
        if window == 1:
            continue;
        new_col_name = 'ROLLING_STD-' + str(window)
        data[new_col_name] = data.groupby([grpby_colname])[attr_name].apply(lambda x: x.rolling(center=False, window = window).std())#.apply(prevent_from_nans)
        
        




class SpecificStandardization():
          
    def __init__(self):
        self.to_minmax_features = []
        self.to_standard_features = []
        self.to_leaveit_features = []
        
    def set_features_to_minmax_scale(self, features):
        self.to_minmax_features = features   
        
    def set_features_to_standard_scale(self, features):
        self.to_standard_features = features
        
    def set_features_to_no_scale(self, features):
        self.to_leaveit_features = features
    
    def print_features(self):
        print(self.to_minmax_features)
        print(self.to_standard_features)
        print(self.to_leaveit_features)     
        
    def is_any_feature_type_set(self):
        if( (len(self.to_minmax_features) == 0) and (len(self.to_standard_features) == 0) ):
            return False
        return True
        
    def get_scaler(self,which_one):
        if which_one == 'standard':
            return self.scaler_standard
        elif which_one == 'minmax':
            return self.scaler_minmax            
        elif which_one == 'target':
            return self.scaler_y            
    
         
    def standardize_all_sepcificly(self,train_x,valid_x,test_x,train_y,valid_y,test_y, is_it_lstm): 
        if( self.is_any_feature_type_set() == False ):
            raise ValueError("Neither minmax features nor standard features were set")
        
        #Instead of [0,1] use [0+E,1-E], where E is very small number
        minmax_min_value = 0.000001
        minmax_max_value = 0.999999        
        
        
        ######################  SPLITTING #########################        
        #Get the colnames for the two group          
        to_standard_colnames = [colname for colname in train_x.columns if colname in self.to_standard_features]        
        to_minmax_colnames = [colname for colname in train_x.columns if colname in self.to_minmax_features]  
        to_leaveit_colnames = [colname for colname in train_x.columns if colname in self.to_leaveit_features]  
       
        #Split to 2 datasets: MinMax Scaling or Standardization
        train_x0 = train_x[to_leaveit_colnames]
        valid_x0 = valid_x[to_leaveit_colnames]
        test_x0 = test_x[to_leaveit_colnames]
        
        train_x1 = train_x[to_standard_colnames]
        valid_x1 = valid_x[to_standard_colnames]
        test_x1 = test_x[to_standard_colnames]
        
        train_x2 = train_x[to_minmax_colnames]
        valid_x2 = valid_x[to_minmax_colnames]
        test_x2 = test_x[to_minmax_colnames]

        
        ########## STANDARD SCALING  ##########
        #Standardize predictor attributes
        scaler_standard = preprocessing.StandardScaler().fit(train_x1)
        train_x1 = scaler_standard.transform(train_x1)
        valid_x1 = scaler_standard.transform(valid_x1)
        test_x1 = scaler_standard.transform(test_x1)
               
        
        ########## MINMAX SCALING  ##########
        #MinMax scale target attributes
        scaler_minmax = preprocessing.MinMaxScaler(feature_range=(minmax_min_value, minmax_max_value)).fit(train_x2)
        train_x2 = scaler_minmax.transform(train_x2)
        valid_x2 = scaler_minmax.transform(valid_x2)
        test_x2 = scaler_minmax.transform(test_x2)
        
        
        #############    TARGET ATTRIBUTES  (Y)  #######################
        #MinMax Scale target attributes
        scaler_y = preprocessing.MinMaxScaler(feature_range=(minmax_min_value, minmax_max_value)).fit(train_y)
        train_y = scaler_y.transform(train_y)
        valid_y = scaler_y.transform(valid_y)
        test_y = scaler_y.transform(test_y)


        #########   CONCATENATION   ##########
        train_x = np.concatenate([train_x0, train_x2, train_x1], axis=1)
        valid_x = np.concatenate([valid_x0, valid_x2, valid_x1], axis=1)
        test_x = np.concatenate([test_x0, test_x2, test_x1], axis=1)
        
        #Create pandas DataFrames because of headers
        new_colnames = np.concatenate((to_leaveit_colnames, to_minmax_colnames, to_standard_colnames))        
        train_x = pd.DataFrame(train_x, columns=new_colnames)
        valid_x = pd.DataFrame(valid_x, columns=new_colnames)
        test_x = pd.DataFrame(test_x, columns=new_colnames)
        
        
        #LSTM and 1Dconv things, would be better not use RETURN  hardcoded
        if is_it_lstm:
            train_x['RETURN'] = train_y.flatten()
            valid_x['RETURN'] = valid_y.flatten()
            test_x['RETURN'] = test_y.flatten()
            
        train_y = pd.DataFrame(train_y, columns=['RETURN'])
        valid_y = pd.DataFrame(valid_y, columns=['RETURN'])
        test_y  = pd.DataFrame(test_y,  columns=['RETURN'])
        
        self.scaler_standard = scaler_standard
        self.scaler_minmax = scaler_minmax   
        self.scaler_y = scaler_y
        return train_x, valid_x, test_x, train_y, valid_y, test_y
        
        
        
        
    
def scaleback_and_compare_target_prediction(scaler,result,target):
    plt.ioff()

    #Scale back predicted value and normalized test_y (for test)
    predicted_unscaled = scaler.inverse_transform(result)
    test_y_unscaled = scaler.inverse_transform(target) 
    
    intervals = []
    full_len = len(result)
    intervals.append(int(full_len/100*30))
    intervals.append(int(full_len/100*60))
    
    fig, axes = plt.subplots(3, 1, figsize=(20, 12))
    axes[0].plot(predicted_unscaled)
    axes[0].plot(test_y_unscaled)
    axes[0].legend(['Predicted (scaled back)', 'Target (scaled back)'], loc='upper left')
    axes[0].set_title('Comparsion on full length')
    
    axes[1].plot(predicted_unscaled[intervals[0]:intervals[0]+500])
    axes[1].plot(test_y_unscaled[intervals[0]:intervals[0]+500])
    axes[1].legend(['Predicted (scaled back)', 'Target (scaled back)'], loc='upper left')
    axes[1].set_title('Comparsion on interval: [' + str(intervals[0]) + ':' + str(intervals[0]+500) + ']')
    
    axes[2].plot(predicted_unscaled[intervals[1]:intervals[1]+500])
    axes[2].plot(test_y_unscaled[intervals[1]:intervals[1]+500])
    axes[2].legend(['Predicted (scaled back)', 'Target (scaled back)'], loc='upper left')
    axes[2].set_title('Comparsion on interval: [' + str(intervals[1]) + ':' + str(intervals[1]+500) + ']')
    
    plt.ion()
    return pd.DataFrame({'Prediction_unscaled':predicted_unscaled.flatten(),'Target_unscaled':test_y_unscaled.flatten()}), fig
    
    
def scaleback_temp(scaler,result,target):

    #Scale back predicted value and normalized test_y (for test)
    predicted_unscaled = scaler.inverse_transform(result)
    print('Shape: ', predicted_unscaled.shape)
    
    return pd.DataFrame({
                        'Prediction_unscaled1':predicted_unscaled[:,0],
                        'Prediction_unscaled2':predicted_unscaled[:,1],
                        'Prediction_unscaled3':predicted_unscaled[:,2],
                        'Prediction_unscaled4':predicted_unscaled[:,3],
                        'Prediction_unscaled5':predicted_unscaled[:,4],
                        'Prediction_unscaled6':predicted_unscaled[:,5],
                        'Prediction_unscaled7':predicted_unscaled[:,6]
                         },index=[i for i in range(predicted_unscaled.shape[0])])
  
  
def compare_target_prediction(result,target):
    mse = mean_squared_error(target,result)
    corr = pearsonr(target,result)
    target_describe = describe(target)
    result_describe = describe(result)
    
    return mse, corr, result_describe, target_describe