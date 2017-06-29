# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import os
import warnings
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.tseries.offsets import *
from pandas.tseries.holiday import USFederalHolidayCalendar as calendar
from os import getcwd
from os.path import dirname
import seaborn as sns
warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
SIZE = 20
plt.rc('font', size=SIZE)        # controls default text sizes
plt.rc('axes', titlesize=35)     # fontsize of the axes title
plt.rc('axes', labelsize=30)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)  # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)  # legend fontsize
plt.rc('figure', titlesize=SIZE)    
plt.rcParams['figure.figsize'] = 16, 12
plt.rcParams['lines.linewidth'] = 4
plt.rcParams['lines.markersize'] = 10
sns.set_style("whitegrid")



#Forecast term
FORECAST_SHIFT = -7



'''
###############################################################################
##################          Loading in data               #####################
###############################################################################
'''
root_path = dirname(dirname(dirname(dirname(getcwd()))))
file_path = os.path.join('dummy','path')
orders = pd.read_csv(file_path, sep='|', index_col=False)

file_path = os.path.join('dummy','name')
products = pd.read_excel(file_path)


'''
#############################################################################################
###########                     ORDERING BY DATE                                #############
#############################################################################################
'''
#Convert Date attributes from Object to Date
orders['REQUESTED_DATE'] = pd.to_datetime(orders['REQUESTED_DATE'])

# Order dataframe by date
orders.sort_index(by='REQUESTED_DATE',inplace=True)


'''
#############################################################################################
###########                     FILTERING                                       #############
#############################################################################################
'''
#Drop unnecessary features
orders.drop(['ORDER_ID', 'LINE_ID', 'CREATION_DATE', 'OPEN_QTY', 'SHIP_TO','LOC_ID'], axis=1,inplace=True)

#Drop records with 0 as QTY
orders = orders[orders['REQUESTED_QTY'] != 0]

#Filtering by date
orders = orders[(orders['REQUESTED_DATE'] < '2016-03-15')]
orders.reset_index(inplace=True, drop=True)


#Save the orders per prod per day for later
df_temp_order_conter = pd.DataFrame()
df_temp_order_conter['ORDERS_THAT_DAY'] = orders.groupby(['REQUESTED_DATE', 'ITEM_ID']).count().iloc[:,0]


#Eliminate items with few occurances
orders = orders.groupby(['REQUESTED_DATE', 'ITEM_ID'], as_index=False).sum() 


#Add the orders per prod per day for later
orders['ORDERS_THAT_DAY'] = df_temp_order_conter['ORDERS_THAT_DAY'].values


items_count = orders['ITEM_ID'].value_counts()                                    #count ITEMs by IDs
not_importan_item_ids = items_count[items_count < 60].index 
length_of_fews = len(orders[orders['ITEM_ID'].isin(not_importan_item_ids)])
orders = orders[~orders['ITEM_ID'].isin(not_importan_item_ids)]                   #drop them
print('I dropped ', len(not_importan_item_ids) ,' items, because lack of enough occurances. That means ', length_of_fews, ' records.')



#Eliminate outliers
orders['Outlier'] = orders.groupby(['ITEM_ID'])['REQUESTED_QTY'].apply(lambda x: abs(x - x.mean()) > 2.9*x.std())
outliers = orders[orders['Outlier'] == True]['REQUESTED_QTY']

idx_of_outliers = outliers.index.tolist()

#Plot sum(quanty) WITH outliers
orders.groupby('REQUESTED_DATE')['REQUESTED_QTY'].sum().plot(kind="line", legend=True, color='red')

orders = orders.loc[~orders.index.isin(idx_of_outliers)]

#Plot sum(quanty) WITHOUT outliers
ax  = orders.groupby('REQUESTED_DATE')['REQUESTED_QTY'].sum().plot(kind="line", legend=True,  label="Quantity WITHOUT outliers")
ax.set_ylabel('REQUESTED_QTY')
orders.reset_index(inplace=True, drop=True)

#Drop Outlier col
orders.drop('Outlier', axis=1, inplace=True)

#del orders_copy
print('I dropped ', len(idx_of_outliers), ' outliers as well.')



'''
#############################################################################################
###########                     SPARSIFY                                        #############
#############################################################################################
'''

#resampling
orders.set_index('REQUESTED_DATE', inplace=True)
orders = orders.groupby(['ITEM_ID']).apply(lambda x: x.resample('D').mean())
orders.reset_index(inplace = True)

orders['REQUESTED_QTY'] = orders['REQUESTED_QTY'].fillna(0.1)
orders['ORDERS_THAT_DAY'] = orders['ORDERS_THAT_DAY'].fillna(0)



'''
#############################################################################################
###########                     CREATING NEW COLUMNS                            #############
#############################################################################################
'''
#Change Nans to given value
def prevent_from_nans(x, sub_value):
    return x if pd.notnull(x) else sub_value

#Log return
def trend(x, mean, var):
    ratio = 0.05
    if (abs(x-mean)<ratio*(var+0.000001)/10):
        return 0
    if (x-mean>ratio*(var+0.000001)/10):
        return 1
    if (x-mean<-ratio*(var+0.000001)/10):
        return 2

#Wrapper function, because Series type is got (whole column and not element by element)
def trend_wrapper(return_col):
    mean = return_col.mean()
    var =  return_col.var()
    return_col = return_col.apply(lambda x: trend(x, mean, var))    
    return return_col


# Independent features: do not depend on other items
orders['YEAR'] = orders['REQUESTED_DATE'].dt.year                       #Add new columns: Year of Date
orders['MONTH'] = orders['REQUESTED_DATE'].dt.month                     #Add new columns: Month of Date
orders['DAY'] = orders['REQUESTED_DATE'].dt.day                         #Add new columns: Day of Date
orders['WEEK'] = orders['REQUESTED_DATE'].dt.week                       #Add new columns: Week of Date
orders['DAY_OF_WEEK'] = orders['REQUESTED_DATE'].dt.dayofweek           #Add new columns: Day of Week
orders['DAY_OF_YEAR'] = orders['REQUESTED_DATE'].dt.dayofyear           #Add new columns: Day of Year
orders['QUARTER_OF_YEAR'] = orders['REQUESTED_DATE'].dt.quarter         #Add new columns: Qaurter of Year
orders['IS_WEEKEND'] = orders['DAY_OF_WEEK'].apply(lambda x: 0 if x < 5 else 1)  #Add new column: Is weekend?
orders['QTY_LOG'] = np.log(orders['REQUESTED_QTY'])                         #Add new column: QTY_LOG

#Sorting
orders.sort(['ITEM_ID','REQUESTED_DATE'], inplace = True)

#Add new column: time gap between REQUESTED_DATEs, still ITEM group specificly
orders['DATE_DELTA'] = orders.groupby(['ITEM_ID'])['REQUESTED_DATE'].apply(lambda x: x.diff().dt.days).apply(prevent_from_nans, args=(1,))

#Add new column: Deltas of date deltas, still ITEM group specificly
orders['DATE_DELTA2'] =  orders.groupby(['ITEM_ID'])['DATE_DELTA'].apply(lambda x: x.diff()).apply(prevent_from_nans, args=(0,))

#Add new column: QTY_DELTA (based on REQUESTED_QTY), still ITEM group specificly
orders['QTY_DELTA'] = orders.groupby(['ITEM_ID'])['REQUESTED_QTY'].apply(lambda x: x.diff()).apply(prevent_from_nans, args=(0,))

#Add new column: QTY_DELTA2 (based on QTY_DELTA), still ITEM group specificly
orders['QTY_DELTA2'] = orders.groupby(['ITEM_ID'])['QTY_DELTA'].apply(lambda x: x.diff()).apply(prevent_from_nans, args=(0,))


#Add new column: RETURN (based on LOG_QTY), still ITEM group specificly
orders['RETURN'] = orders.groupby(['ITEM_ID'])['QTY_LOG'].apply(lambda x: x.shift(FORECAST_SHIFT) - x)


#Add new column: QTY_PRED (based on REQUESTED_QTY), still ITEM group specificly
orders['QTY_PRED'] = orders.groupby(['ITEM_ID'])['REQUESTED_QTY'].apply(lambda x: x.shift(FORECAST_SHIFT))



#Eliminate NaNs by dropping the last row of each group
orders.dropna(inplace=True)
orders.reset_index(inplace=True, drop=True)



#Add new column: MOVEMENT (based on RETURN), still ITEM group specificly
orders['MOVEMENT'] = orders.groupby(['ITEM_ID'])['RETURN'].apply(trend_wrapper)
orders['MOVEMENT'].value_counts()

'''
#############################################################################################
###########                    DO SOMETHING WITH THE ZERO PERIODS               #############
#############################################################################################
'''
#Add new column: sum of zeros until that day
orders['IS_ZERO'] = orders['REQUESTED_QTY'] == 0.1
orders['ZEROS_CUMSUM'] = orders.groupby('ITEM_ID')['IS_ZERO'].apply(lambda x: x.cumsum())

#Add new column: zeros number/observation nubmer until that point
orders['ZERO_FULL_RATIO'] = orders.groupby('ITEM_ID')['ZEROS_CUMSUM'].apply(lambda x: x/range(1,len(x)+1)) 

#Add new column: Days passed since last zero value: y * (y.groupby((y != y.shift()).cumsum()).cumcount() + 1)
#It is not a problem that its not done in groupedby PROD_GRP way, because first value is always 0
orders['ZERO_QTY_SEQUENCE'] = orders['IS_ZERO'] * (orders.groupby((orders['IS_ZERO'] != orders['IS_ZERO'].shift()).cumsum()).cumcount() + 1)


#Add new column: longest zero period until that point
def cumcount_max_zeroseqence(group):
    values = group['ZERO_QTY_SEQUENCE'].values
    max_untilthat = values[0]
    ls_maxes = []
    for val in values:
        if val > max_untilthat:
            max_untilthat = val
        ls_maxes.append(max_untilthat)
    return pd.Series(ls_maxes)
orders['MAX_ZERO_SEQUENCE'] = orders.groupby('ITEM_ID').apply(cumcount_max_zeroseqence).values



#Add new column: AVG zero peroid length until that point
def cumcount_mean_zeroseqence(group):
    values = group['ZERO_QTY_SEQUENCE'].values
    ls_means = []
    for i, val in enumerate(values):
        mean_actual = np.mean(values[:i+1])
        ls_means.append(mean_actual)
    return pd.Series(ls_means)
orders['MEAN_OF_ZERO_SEQ'] = orders.groupby('ITEM_ID').apply(cumcount_mean_zeroseqence).values




'''
#############################################################################################
###########                   NEW COLUMNS: DATE SPECIFIC ATTRS                  #############
#############################################################################################
'''

def create_new_column(new_col_name, orders, REQUESTED_DATE):
    if new_col_name == 'NEXT_HOLIDAY':
        #Holiday stuffs
        def next_holi(holi_dates, actual_day):
            delta = holi_dates[actual_day:][0] - actual_day
            return int(delta.days)
        
        min_date = orders[REQUESTED_DATE].min()
        max_date = orders[REQUESTED_DATE].max()
    
        cal = calendar()
        holidays = cal.holidays()
        
        se_holidays = holidays.to_series()
        se_holidays_relevant = se_holidays[min_date:se_holidays[max_date:][0]] #relevant_dates plus next holiday
        
        
        df_unique_dates = pd.DataFrame(orders['REQUESTED_DATE'].unique(), columns=['REQUESTED_DATE'])
        df_unique_dates['NEXT_HOLIDAY'] = df_unique_dates['REQUESTED_DATE'].map(lambda x: next_holi(se_holidays_relevant, x))
        
        
        result_column = pd.merge(orders, df_unique_dates, how='left', on=['REQUESTED_DATE'])['NEXT_HOLIDAY']        
        return result_column
        
              
        
    elif new_col_name == 'PREV_HOLIDAY':
        def prev_holi(holi_dates, actual_day):
            delta =  actual_day - holi_dates[holi_dates < actual_day][-1]
            return int(delta.days)
        
        min_date = orders[REQUESTED_DATE].min()
        max_date = orders[REQUESTED_DATE].max()
    
        cal = calendar()
        holidays = cal.holidays()
        
        se_holidays = holidays.to_series()
        se_holidays_relevant = se_holidays[se_holidays[se_holidays < min_date][-1] : max_date]     
        
        df_unique_dates = pd.DataFrame(orders['REQUESTED_DATE'].unique(), columns=['REQUESTED_DATE'])
        df_unique_dates['PREV_HOLIDAY'] = df_unique_dates['REQUESTED_DATE'].map(lambda x: prev_holi(se_holidays_relevant, x))

        result_column = pd.merge(orders, df_unique_dates, how='left', on=['REQUESTED_DATE'])['PREV_HOLIDAY']        
        return result_column        
        
        
    elif new_col_name == 'NEXT_MNTH_END':
        df_unique_dates = pd.DataFrame(orders['REQUESTED_DATE'].unique(), columns=['REQUESTED_DATE'])
        df_unique_dates['NEXT_MNTH_END'] = ((df_unique_dates['REQUESTED_DATE'] + MonthEnd()) - df_unique_dates['REQUESTED_DATE']).dt.days
        result_column = pd.merge(orders, df_unique_dates, how='left', on=['REQUESTED_DATE'])['NEXT_MNTH_END']        
        return result_column       
       
    elif new_col_name == 'NEXT_QRT_END':
        df_unique_dates = pd.DataFrame(orders['REQUESTED_DATE'].unique(), columns=['REQUESTED_DATE'])
        df_unique_dates['NEXT_QRT_END'] = ((df_unique_dates['REQUESTED_DATE'] + QuarterEnd()) - df_unique_dates['REQUESTED_DATE']).dt.days
        result_column = pd.merge(orders, df_unique_dates, how='left', on=['REQUESTED_DATE'])['NEXT_QRT_END']        
        return result_column           

    elif new_col_name == 'DAY_OF_QRT':
        df_unique_dates = pd.DataFrame(orders['REQUESTED_DATE'].unique(), columns=['REQUESTED_DATE'])
        df_unique_dates['DAY_OF_QRT'] = (df_unique_dates['REQUESTED_DATE'] - (df_unique_dates['REQUESTED_DATE'] - QuarterEnd())).dt.days
        result_column = pd.merge(orders, df_unique_dates, how='left', on=['REQUESTED_DATE'])['DAY_OF_QRT']        
        return result_column           

#TODO: check why does it need here?
orders.reset_index(inplace=True, drop=True)   
orders['NEXT_MNTH_END'] = create_new_column('NEXT_MNTH_END', orders, 'REQUESTED_DATE')       
orders['NEXT_QRT_END'] = create_new_column('NEXT_QRT_END', orders, 'REQUESTED_DATE')     
orders['DAY_OF_QRT'] = create_new_column('DAY_OF_QRT', orders, 'REQUESTED_DATE')        
orders['NEXT_HOLIDAY'] = create_new_column('NEXT_HOLIDAY', orders, 'REQUESTED_DATE')       
orders['PREV_HOLIDAY'] = create_new_column('PREV_HOLIDAY', orders, 'REQUESTED_DATE')


#Info
orders.info()

#Check Nans
print((pd.isnull(orders)).any())
print('True means Nan values are contained!!!')

#FIXME:
#Save to disk
#with open('generated_data/data_prepared_itemized.pickle', 'wb') as f:
#    pickle.dump(orders, f, -1)



