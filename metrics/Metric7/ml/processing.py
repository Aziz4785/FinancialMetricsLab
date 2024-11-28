import pandas as pd
import random
from datetime import datetime, timedelta
import sys
import os
from ..utils import *
from ...common import *


"""
to run : to run : py -m metrics.Metric6.ml.processing
if there is a lot of None values at the end, re-run this script because sometimes the problem comes from the API
"""



df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric6/ml/cleaned_training_data.csv', parse_dates=['date'])
#df = df.sample(frac=0.1)

stocks = df['symbol'].unique()
print(f"number of unique stocks : {len(stocks)}")
if (len(stocks) <3):
    print("too few stocks ... cannot perform analysis on it")
    exit()

historical_data_for_stock, market_cap_dict, income_dict, hist_data_df_for_stock,balance_dict,cashflow_dict,estimations_dict,sector_dict = fetch_stock_data(stocks)
sma_10d_dict,sma_50w_dict,sma_100d_dict,sma_200d_dict,sma_50d_dict,_,_,std_10d_dict = fetch_data_for_ml(stocks)

df['sma_10d'] = df.apply(lambda row: extract_SMA(row['date'],sma_10d_dict[row['symbol']]), axis=1)
print("SMA 10D done !")
df['sma_50w'] = df.apply(lambda row: extract_SMA(row['date'],sma_50w_dict[row['symbol']]), axis=1)
print("SMA 50W done !")
df['sma_100d'] = df.apply(lambda row: extract_SMA( row['date'],  sma_100d_dict[row['symbol']]), axis=1)
df['sma_200d'] = df.apply(lambda row: extract_SMA( row['date'], sma_200d_dict[row['symbol']]), axis=1)
print("SMA 100d done !")
df['sma_50d'] = df.apply(lambda row: extract_SMA( row['date'],  sma_50d_dict[row['symbol']]), axis=1)

df['std_10d'] = df.apply(lambda row: extract_STD( row['date'],  std_10d_dict[row['symbol']]), axis=1)
print("-------SMA DONE------------")

df = pd.concat([
    df,
    df.apply(lambda row: pd.Series(
        extract_multiple_SMAs(row['date'], sma_10d_dict.get(row['symbol']))
    ), axis=1)
], axis=1)

df['month'] = df['date'].dt.month

print("-------SMA X MONTHS AGO DONE-------------")
df['RnD_expenses'] = df.apply(lambda row: extract_Field(row['date'], income_dict[row['symbol']],'researchAndDevelopmentExpenses'), axis=1)
df['revenues']=df.apply(lambda row: extract_Field(row['date'], income_dict[row['symbol']],'revenue'), axis=1)
df['GP']=df.apply(lambda row: extract_Field(row['date'], income_dict[row['symbol']],'grossProfit'), axis=1)
df['netIncome']=df.apply(lambda row: extract_Field(row['date'], income_dict[row['symbol']],'netIncome'), axis=1)

df['dividendsPaid']=df.apply(lambda row: extract_Field(row['date'], cashflow_dict[row['symbol']],'dividendsPaid'), axis=1)



df['sector']=df.apply(lambda row: sector_dict[row['symbol']], axis=1)
#df = pd.get_dummies(df, columns=['sector']) #one hot encoding

results = df.apply(
    lambda row: extract_EBITDA(row['date'], income_dict[row['symbol']]), 
    axis=1
)
df['ebitda'], df['income_lag_days'] = zip(*[
    (None, None) if result is None else result 
    for result in results
])
print("df .columns = ",df.columns)
df['marketCap'] = df.apply(lambda row: extract_market_cap(row['date'], market_cap_dict[row['symbol']]), axis=1)

results2 = df.apply(
    lambda row: extract_total_debt(row['date'], balance_dict[row['symbol']]), 
    axis=1
)
df['total_debt'], df['balance_lag_days'] = zip(*[
    (None, None) if result is None else result 
    for result in results2
])

df['other_expenses']=df.apply(lambda row: extract_Field(row['date'], income_dict[row['symbol']],'otherExpenses'), axis=1)

df['cashcasheq']=df.apply(lambda row: extract_Field(row['date'], balance_dict[row['symbol']],'cashAndCashEquivalents'), axis=1)
df['netDebt']=df.apply(lambda row: extract_Field(row['date'], balance_dict[row['symbol']],'netDebt'), axis=1)
results3 = df.apply(
    lambda row: extract_current_and_future_estim_eps(row['date'], estimations_dict[row['symbol']]), 
    axis=1
)
df['curr_est_eps'], df['future_est_eps'] = zip(*[
    (None, None) if result is None else result 
    for result in results3
])

results4 = df.apply(
    lambda row: extract_current_and_future_estim_REV(row['date'], estimations_dict[row['symbol']]), 
    axis=1
)
df['curr_est_rev'], df['future_est_rev'] = zip(*[
    (None, None) if result is None else result 
    for result in results4
])


df['EV'] = df.apply(lambda row: row['marketCap']+row['total_debt']-row['cashcasheq'], axis=1)

print("-------fundamental metrics done----------")
results_max4M = df.apply(lambda row: extract_max_price_in_range((pd.to_datetime(row['date']) - pd.DateOffset(months=4)).date(),row['date'],hist_data_df_for_stock[row['symbol']],return_date=True), axis=1)
df['max_in_4M'], df['max4M_date'] = zip(*[(None, None) if result is None else result for result in results_max4M])
df['max4M_lag']=df.apply(lambda row: (pd.to_datetime(row['date']) - pd.to_datetime(row['max4M_date'])).days, axis=1)
results_min4M = df.apply(lambda row: extract_min_price_in_range((pd.to_datetime(row['date']) - pd.DateOffset(months=4)).date(),row['date'],hist_data_df_for_stock[row['symbol']],return_date=True), axis=1)
df['min_in_4M'], df['min4M_date'] = zip(*[(None, None) if result is None else result for result in results_min4M])
df['min4M_lag']=df.apply(lambda row: (pd.to_datetime(row['date']) - pd.to_datetime(row['min4M_date'])).days, axis=1)
df = df.drop('min4M_date', axis=1)
df = df.drop('max4M_date', axis=1)
df['max_minus_min']=df.apply(lambda row: None if (row['max_in_4M'] is None or row['min_in_4M'] is None) else row['max_in_4M'] - row['min_in_4M'], axis=1)
df['var_price_to_minmax'] = df.apply(lambda row: 
    None if row['max_minus_min'] is None or row['max_minus_min'] == 0 
    else (row['price'] - row['min_in_4M'])/row['max_minus_min'], 
    axis=1)
df['maxPercen_4M'] = df.apply(lambda row: 
    None if row['price'] is None or row['price'] == 0 
    else (row['max_in_4M'] - row['price'])/row['price'], 
    axis=1)


results_max1M = df.apply(lambda row: extract_max_price_in_range((pd.to_datetime(row['date']) - pd.DateOffset(months=1)).date(),row['date'],hist_data_df_for_stock[row['symbol']],return_date=True), axis=1)
df['max_in_1M'], df['max1M_date'] = zip(*[(None, None) if result is None else result for result in results_max1M])
df['max1M_lag']=df.apply(lambda row: (pd.to_datetime(row['date']) - pd.to_datetime(row['max1M_date'])).days, axis=1)
results_min1M = df.apply(lambda row: extract_min_price_in_range((pd.to_datetime(row['date']) - pd.DateOffset(months=1)).date(),row['date'],hist_data_df_for_stock[row['symbol']],return_date=True), axis=1)
df['min_in_1M'], df['min1M_date'] = zip(*[(None, None) if result is None else result for result in results_min1M])
df['min1M_lag']=df.apply(lambda row: (pd.to_datetime(row['date']) - pd.to_datetime(row['min1M_date'])).days, axis=1)
df = df.drop('min1M_date', axis=1)
df = df.drop('max1M_date', axis=1)
df['max_minus_min1M']=df.apply(lambda row: None if (row['max_in_1M'] is None or row['min_in_1M'] is None) else row['max_in_1M'] - row['min_in_1M'], axis=1)
df['var_price_to_minmax1M'] = df.apply(lambda row: 
    safe_divide(safe_subtract(row['price'] , row['min_in_1M']),row['max_minus_min1M']), 
    axis=1)
df['maxPercen_1M'] = df.apply(lambda row: 
    safe_divide(safe_subtract(row['max_in_1M'], row['price']),row['price']), 
    axis=1)


results_max2W = df.apply(lambda row: extract_max_price_in_range((pd.to_datetime(row['date']) - pd.DateOffset(weeks=2)).date(),row['date'],hist_data_df_for_stock[row['symbol']],return_date=True), axis=1)
df['max_in_2W'], df['max2W_date'] = zip(*[(None, None) if result is None else result for result in results_max2W])
df['max2W_lag']=df.apply(lambda row: (pd.to_datetime(row['date']) - pd.to_datetime(row['max2W_date'])).days, axis=1)
results_min2W = df.apply(lambda row: extract_min_price_in_range((pd.to_datetime(row['date']) - pd.DateOffset(weeks=2)).date(),row['date'],hist_data_df_for_stock[row['symbol']],return_date=True), axis=1)
df['min_in_2W'], df['min2W_date'] = zip(*[(None, None) if result is None else result for result in results_min2W])
df['min2W_lag']=df.apply(lambda row: (pd.to_datetime(row['date']) - pd.to_datetime(row['min2W_date'])).days, axis=1)
df = df.drop('min2W_date', axis=1)
df = df.drop('max2W_date', axis=1)
df['max_minus_min2W']=df.apply(lambda row: None if (row['max_in_2W'] is None or row['min_in_2W'] is None) else row['max_in_2W'] - row['min_in_2W'], axis=1)
df['var_price_to_minmax2W'] = df.apply(lambda row: 
    safe_divide(safe_subtract(row['price'], row['min_in_2W']),row['max_minus_min2W']), 
    axis=1)
df['maxPercen_2W'] = df.apply(lambda row: 
    None if row['price'] is None or row['price'] == 0 
    else (row['max_in_2W'] - row['price'])/row['price'], 
    axis=1)


results_max8M = df.apply(lambda row: extract_max_price_in_range((pd.to_datetime(row['date']) - pd.DateOffset(months=8)).date(),row['date'],hist_data_df_for_stock[row['symbol']],return_date=True), axis=1)
df['max_in_8M'], df['max8M_date'] = zip(*[(None, None) if result is None else result for result in results_max8M])
df['max8M_lag']=df.apply(lambda row: None if row['max8M_date'] is None else (pd.to_datetime(row['date']) - pd.to_datetime(row['max8M_date'])).days, axis=1)
df['dist_max8M_4M'] = df.apply(lambda row: 
    row['max8M_lag'] - row['max4M_lag'], 
    axis=1)
results_min8M = df.apply(lambda row: extract_min_price_in_range((pd.to_datetime(row['date']) - pd.DateOffset(months=8)).date(),row['date'],hist_data_df_for_stock[row['symbol']],return_date=True), axis=1)
df['min_in_8M'], df['min8M_date'] = zip(*[(None, None) if result is None else result for result in results_min8M])
df['min8M_lag']=df.apply(lambda row: None if row['min8M_date'] is None else (pd.to_datetime(row['date']) - pd.to_datetime(row['min8M_date'])).days, axis=1)
df['dist_min8M_4M'] = df.apply(lambda row: 
    row['min8M_lag'] - row['min4M_lag'], 
    axis=1)

df = df.drop('min8M_date', axis=1)
df = df.drop('max8M_date', axis=1)

df['max_minus_min8M']=df.apply(lambda row: None if row['max_in_8M'] is None or row['min_in_8M'] is None else row['max_in_8M'] - row['min_in_8M'], axis=1)
df['var_price_minmax8M'] = df.apply(lambda row: 
    safe_divide(safe_subtract(row['price'],row['min_in_8M']),row['max_minus_min8M']), 
    axis=1)
df['maxPercen_8M'] = df.apply(lambda row: 
    safe_divide(safe_subtract(row['max_in_8M'], row['price']),row['price']), 
    axis=1)

# print("df.columns : ",df.columns)
# print([f"'{col}'" for col in df.columns]) 
# print(df['ebitda'].head())

df['ebitdaMargin'] = df.apply(
    lambda row: safe_divide(row['ebitda'], row['revenues']), axis=1
) #By default, df.apply() applies the function to each column (axis=0), not each row.

df['Spread1Mby8M'] = df.apply(lambda row: 
    safe_divide(row['max_minus_min1M'],row['max_minus_min8M']), 
    axis=1)
df['Spread4Mby8M'] = df.apply(lambda row: 
    safe_divide(row['max_minus_min'],row['max_minus_min8M']), 
    axis=1)
df['Spread1Mby4M'] = df.apply(lambda row: 
    safe_divide(row['max_minus_min1M'],row['max_minus_min']), 
    axis=1)

print("--------max and low done------------")
df['debtToPrice'] = df.apply(lambda row: 
    safe_divide(row['total_debt'],row['marketCap']), 
    axis=1)

df['markRevRatio'] = df.apply(lambda row: 
    None if row['revenues'] is None or row['revenues'] == 0 
    else row['marketCap']/row['revenues'], 
    axis=1)




df['eps_growth'] = df.apply(lambda row: 
    None if row['curr_est_eps'] is None or row['curr_est_eps'] == 0 
    else safe_subtract(row['future_est_eps'],row['curr_est_eps'])/row['curr_est_eps'], 
    axis=1)

df = df.drop('future_est_eps', axis=1)

df['pe']=df.apply(lambda row: 
    None if row['curr_est_eps'] is None or row['curr_est_eps'] == 0 
    else (row['price'])/row['curr_est_eps'], 
    axis=1)

df['peg'] = df.apply(lambda row: 
    None if row['eps_growth'] is None or row['eps_growth'] == 0 
    else row['pe']/row['eps_growth'], 
    axis=1)

df['PS_to_PEG'] = df.apply(lambda row: 
    None if row['markRevRatio'] is None or row['markRevRatio'] == 0 
    else row['peg']/row['markRevRatio'], 
    axis=1)

df['netDebtToPrice'] = df.apply(lambda row: 
    None if row['marketCap'] is None or row['marketCap'] == 0 
    else row['netDebt']/row['marketCap'], 
    axis=1)

df['dividend_payout_ratio'] = df.apply(lambda row: 
    None if row['netIncome'] is None or row['netIncome'] == 0 
    else row['dividendsPaid']/row['netIncome'], 
    axis=1)

df['EVEbitdaRatio'] = df.apply(lambda row: 
    None if row['ebitda'] is None or row['ebitda'] == 0 
    else row['EV']/row['ebitda'], 
    axis=1)

df['EVGP'] = df.apply(lambda row: 
    safe_divide(row['EV'],row['GP']), 
    axis=1)


df['EVRevenues'] = df.apply(lambda row: 
    None if row['revenues'] is None or row['revenues'] == 0 
    else row['EV']/row['revenues'], 
    axis=1)

df['fwdPriceTosale'] = df.apply(lambda row: 
    None if row['future_est_rev'] is None or row['future_est_rev'] == 0 
    else row['marketCap']/row['future_est_rev'], 
    axis=1)

df['fwdPriceTosale_diff'] = df.apply(lambda row: 
    safe_subtract(row['fwdPriceTosale'],row['markRevRatio']), 
    axis=1)

"""df['forward_pe'] = df.apply(lambda row: 
    None if row['future_est_eps'] is None or row['future_est_eps'] == 0 
    else row['price']/row['future_est_eps'], 
    axis=1)"""

"""df['var_sma50W'] = df.apply(lambda row: 
    None if row['sma_50w'] is None or row['sma_50w'] == 0 
    else (row['price'] - row['sma_50w'])/row['sma_50w'], 
    axis=1)"""
df['var_sma50D'] = df.apply(lambda row: 
    safe_divide(safe_subtract(row['price'], row['sma_50d']),row['sma_50d']), 
    axis=1)
df['var_sma100D'] = df.apply(lambda row: 
    safe_divide(safe_subtract(row['price'] ,row['sma_100d']),row['sma_100d']), 
    axis=1)
df['var_sma10D'] = df.apply(lambda row: 
    safe_divide(safe_subtract(row['price'] , row['sma_10d']),row['sma_10d']), 
    axis=1)

df['var_sma10D_50D'] = df.apply(lambda row: 
    safe_divide(safe_subtract(row['sma_10d'] , row['sma_50d']),row['sma_50d']), 
    axis=1)

df['var_sma50D_100D'] = df.apply(lambda row: 
    safe_divide((row['sma_50d'] - row['sma_100d']),row['sma_100d']), 
    axis=1)

df['var_sma50D_200D'] = df.apply(lambda row: 
    safe_divide(safe_subtract(row['sma_50d'],row['sma_200d']),row['sma_200d']), 
    axis=1)

df['var_sma10D_100D'] = df.apply(lambda row: 
    None if row['sma_100d'] is None or row['sma_100d'] == 0 
    else safe_subtract(row['sma_10d'], row['sma_100d'])/row['sma_100d'], 
    axis=1)

df['var_sma10D_200D'] = df.apply(lambda row: 
    None if row['sma_200d'] is None or row['sma_200d'] == 0 
    else safe_subtract(row['sma_10d'] ,row['sma_200d'])/row['sma_200d'], 
    axis=1)


df['relative_std'] = df.apply(lambda row: 
    None if row['price'] is None or row['price'] == 0 
    else row['std_10d']/row['price'], 
    axis=1)

print("-------var sma done----------")

df['deriv_1w'] = df.apply(lambda row: 
    (row['price'] - row['sma_10d_1weeks_ago'])/0.25, 
    axis=1)

df['deriv_1m'] = df.apply(lambda row: 
    (row['price'] - row['sma_10d_1months_ago'])/1, 
    axis=1)

df['deriv_2m'] = df.apply(lambda row: 
    (row['price'] - row['sma_10d_2months_ago'])/2, 
    axis=1)

df['deriv_3m'] = df.apply(lambda row: 
    (row['price'] - row['sma_10d_3months_ago'])/3, 
    axis=1)

df['deriv_4m'] = df.apply(lambda row: 
    safe_subtract(row['price'], row['sma_10d_4months_ago'])/4, 
    axis=1)

df['deriv_5m'] = df.apply(lambda row: 
    safe_subtract(row['price'],row['sma_10d_5months_ago'])/5, 
    axis=1)

df['deriv_6m'] = df.apply(lambda row: 
    safe_subtract(row['price'], row['sma_10d_6months_ago'])/6, 
    axis=1)

df['deriv_max4M'] = df.apply(lambda row: None if row['max4M_lag']is None or row['max4M_lag']==0 else
    safe_subtract(row['price'] ,row['max_in_4M'])/row['max4M_lag'], 
    axis=1)
df['deriv_min4M'] = df.apply(lambda row: None if row['min4M_lag']is None or row['min4M_lag']==0 else
    safe_subtract(row['price'] , row['min_in_4M'])/row['min4M_lag'], 
    axis=1)

df['deriv_max8M'] = df.apply(lambda row: None if row['max8M_lag']is None or row['max8M_lag']==0 else
    safe_subtract(row['price'] ,row['max_in_8M'])/row['max8M_lag'], 
    axis=1)

df['deriv_min8M'] = df.apply(lambda row: None if row['min8M_lag'] is None or row['min8M_lag']==0 else
    safe_subtract(row['price'],row['min_in_8M'])/row['min8M_lag'], 
    axis=1)

df['deriv_1w1m_growth'] = df.apply(lambda row: 
    None if row['deriv_1m'] is None or row['deriv_1m'] == 0 
    else safe_subtract(row['deriv_1w'] , row['deriv_1m'])/row['deriv_1m'], 
    axis=1)

df['deriv_1m2m_growth'] = df.apply(lambda row: 
    None if row['deriv_2m'] is None or row['deriv_2m'] == 0 
    else safe_subtract(row['deriv_1m'],row['deriv_2m'])/row['deriv_2m'], 
    axis=1)

df['deriv_3m4m_growth'] = df.apply(lambda row: 
    None if row['deriv_4m'] is None or row['deriv_4m'] == 0 
    else safe_subtract(row['deriv_3m'],row['deriv_4m'])/row['deriv_4m'], 
    axis=1)

df['1W_return'] = df.apply(lambda row: 
    None if row['sma_10d_1weeks_ago'] is None or row['sma_10d_1weeks_ago'] == 0 
    else safe_subtract(row['price'],row['sma_10d_1weeks_ago'])/row['sma_10d_1weeks_ago'], 
    axis=1)

df['price_to_SMA10d_1W'] = df.apply(lambda row: 
    None if row['sma_10d_1weeks_ago'] is None or row['sma_10d_1weeks_ago'] == 0 
    else row['price']/row['sma_10d_1weeks_ago'], 
    axis=1)

df['price_to_SMA10d_5M'] = df.apply(lambda row: 
    None if row['sma_10d_5months_ago'] is None or row['sma_10d_5months_ago'] == 0 
    else row['price']/row['sma_10d_5months_ago'], 
    axis=1)

df['1M_return'] = df.apply(lambda row: 
    None if row['sma_10d_1months_ago'] is None or row['sma_10d_1months_ago'] == 0 
    else safe_subtract(row['price'],row['sma_10d_1months_ago'])/row['sma_10d_1months_ago'], 
    axis=1)

df['2M_return'] = df.apply(lambda row: 
    None if row['sma_10d_2months_ago'] is None or row['sma_10d_2months_ago'] == 0 
    else safe_subtract(row['price'],row['sma_10d_2months_ago'])/row['sma_10d_2months_ago'], 
    axis=1)

df['3M_return'] = df.apply(lambda row: 
    None if row['sma_10d_3months_ago'] is None or row['sma_10d_3months_ago'] == 0 
    else safe_subtract(row['price'],row['sma_10d_3months_ago'])/row['sma_10d_3months_ago'], 
    axis=1)

df['4M_return'] = df.apply(lambda row: 
    None if row['sma_10d_4months_ago'] is None or row['sma_10d_4months_ago'] == 0 
    else safe_subtract(row['price'],row['sma_10d_4months_ago'])/row['sma_10d_4months_ago'], 
    axis=1)

df['5M_return'] = df.apply(lambda row: 
    None if row['sma_10d_5months_ago'] is None or row['sma_10d_5months_ago'] == 0 
    else safe_subtract(row['price'],row['sma_10d_5months_ago'])/row['sma_10d_5months_ago'], 
    axis=1)

df['6M_return'] = df.apply(lambda row: 
    None if row['sma_10d_6months_ago'] is None or row['sma_10d_6months_ago'] == 0 
    else safe_subtract(row['price'],row['sma_10d_6months_ago'])/row['sma_10d_6months_ago'], 
    axis=1)

df['7M_return'] = df.apply(lambda row: 
    None if row['sma_10d_7months_ago'] is None or row['sma_10d_7months_ago'] == 0 
    else safe_subtract(row['price'],row['sma_10d_7months_ago'])/row['sma_10d_7months_ago'], 
    axis=1)

df['1y_return'] = df.apply(lambda row: 
    None if row['sma_10d_11months_ago'] is None or row['sma_10d_11months_ago'] == 0 
    else safe_subtract(row['price'],row['sma_10d_11months_ago'])/row['sma_10d_11months_ago'], 
    axis=1)
print("-------------------")

df['1M_1W_growth']= df.apply(lambda row: 
    None if row['sma_10d_1months_ago'] is None or row['sma_10d_1months_ago'] == 0 
    else safe_subtract(row['sma_10d_1weeks_ago'],row['sma_10d_1months_ago'])/row['sma_10d_1months_ago'], 
    axis=1)

df['2M_1W_growth']= df.apply(lambda row: 
    None if row['sma_10d_2months_ago'] is None or row['sma_10d_2months_ago'] == 0 
    else safe_subtract(row['sma_10d_1weeks_ago'],row['sma_10d_2months_ago'])/row['sma_10d_2months_ago'], 
    axis=1)

df['1M1W_2M1W_growth']= df.apply(lambda row: 
    None if row['2M_1W_growth'] is None or row['2M_1W_growth'] == 0 
    else safe_subtract(row['1M_1W_growth'],row['2M_1W_growth'])/row['2M_1W_growth'], 
    axis=1)

df['2M_1M_growth']= df.apply(lambda row: 
    None if row['sma_10d_2months_ago'] is None or row['sma_10d_2months_ago'] == 0 
    else safe_subtract(row['sma_10d_1months_ago'], row['sma_10d_2months_ago'])/row['sma_10d_2months_ago'], 
    axis=1)

df['3M_2M_growth']= df.apply(lambda row: 
    None if row['sma_10d_3months_ago'] is None or row['sma_10d_3months_ago'] == 0 
    else safe_subtract(row['sma_10d_2months_ago'], row['sma_10d_3months_ago'])/row['sma_10d_3months_ago'], 
    axis=1)

df['4M_3M_growth']= df.apply(lambda row: 
    None if row['sma_10d_4months_ago'] is None or row['sma_10d_4months_ago'] == 0 
    else safe_subtract(row['sma_10d_3months_ago'], row['sma_10d_4months_ago'])/row['sma_10d_4months_ago'], 
    axis=1)

df['1Y_6M_growth']= df.apply(lambda row: 
    None if row['sma_10d_11months_ago'] is None or row['sma_10d_11months_ago'] == 0 
    else safe_subtract(row['sma_10d_6months_ago'],row['sma_10d_11months_ago'])/row['sma_10d_11months_ago'], 
    axis=1)



print("length of df  : ",len(df))
null_counts = df.isnull().sum()
pd.set_option('display.max_rows', None)
print(null_counts)
"""none_counts = {
    'debtToPrice': df['debtToPrice'].isnull().sum(),
    '4M_3M_growth': df['4M_3M_growth'].isnull().sum(),
    '4M_return': df['4M_return'].isnull().sum(),
    'sma_100d':df['sma_100d'].isnull().sum(),
    'sma_50d':df['sma_50d'].isnull().sum(),
    'sma_10d_5months_ago':df['sma_50d'].isnull().sum(),
    'curr_est_eps':df['curr_est_eps'].isnull().sum(),
    'future_est_eps':df['future_est_eps'].isnull().sum(),
    'sma_10d_4months_ago':df['sma_10d_4months_ago'].isnull().sum(),
    'min_in_8M':df['min_in_8M'].isnull().sum(),
    'min_in_4M':df['min_in_4M'].isnull().sum(),
    'sma_10d_1weeks_ago': df['sma_10d_1weeks_ago'].isnull().sum(),
    'price': df['price'].isnull().sum(),
    'sma_10d': df['sma_10d'].isnull().sum(),
    'sma_50w': df['sma_10d'].isnull().sum(),
    'sma_200d': df['sma_200d'].isnull().sum(),
    'std_10d': df['std_10d'].isnull().sum(),
    'sma_10d_11months_ago': df['sma_10d_11months_ago'].isnull().sum(),
    'sma_10d_1months_ago': df['sma_10d_1months_ago'].isnull().sum(),
    'sma_10d_2months_ago': df['sma_10d_2months_ago'].isnull().sum(),
    'sma_10d_3months_ago': df['sma_10d_3months_ago'].isnull().sum(),
    'sma_10d_6months_ago': df['sma_10d_6months_ago'].isnull().sum(),
    'sma_10d_7months_ago': df['sma_10d_7months_ago'].isnull().sum(),
    'RnD_expenses':  df['RnD_expenses'].isnull().sum(),
    'revenues':  df['revenues'].isnull().sum(),
    'GP':  df['GP'].isnull().sum(),
    'ebitda':  df['ebitda'].isnull().sum(),
    'income_lag_days':  df['income_lag_days'].isnull().sum(),
    'marketCap':  df['marketCap'].isnull().sum(),
    'total_debt':  df['total_debt'].isnull().sum(),
    'balance_lag_days':  df['balance_lag_days'].isnull().sum(),
    'cashcasheq':  df['cashcasheq'].isnull().sum(),
    'netDebt':  df['netDebt'].isnull().sum(),
    'curr_est_eps':  df['curr_est_eps'].isnull().sum(),
    'future_est_eps':  df['future_est_eps'].isnull().sum(),
    'curr_est_rev':  df['curr_est_rev'].isnull().sum(),
    'future_est_rev':  df['future_est_rev'].isnull().sum(),
    'max_in_4M':  df['max_in_4M'].isnull().sum(),
    'EV':  df['EV'].isnull().sum(),
    'min_in_4M','max_minus_min','var_price_to_minmax',maxPercen_4M,max_in_8M,min_in_8M,max_minus_min8M,var_price_minmax8M,maxPercen_8M,debtToPrice,markRevRatio,eps_growth,pe,peg,netDebtToPrice,EVEbitdaRatio,EVGP,fwdPriceTosale,var_sma50W,var_sma50D,var_sma100D,var_sma10D,var_sma10D_50D,var_sma50D_100D,var_sma50D_200D,var_sma10D_100D,var_sma10D_200D,relative_std,deriv_1w,deriv_1m,deriv_2m,deriv_3m,deriv_4m,deriv_1w1m_growth,deriv_1m2m_growth,1W_return,1M_return,2M_return,3M_return,4M_return,5M_return,6M_return,7M_return,1y_return,1M_1W_growth,2M_1W_growth,
}


for column, count in none_counts.items():
    print(f"Number of None values in {column}: {count}")"""
    

total_rows = len(df)

# drop columns with a lot of nulls
null_percentages = (df.isnull().sum() / total_rows) * 100
columns_to_drop = null_percentages[null_percentages > 10].index
df = df.drop(columns=columns_to_drop)
print(f"Columns dropped: {list(columns_to_drop)}")

df = df.dropna()

print("length of df before engineer_stock_features : ",len(df))
df = engineer_stock_features(df)
#df = calculate_sector_relatives(df, 'sector')
df = df.dropna()
print("length of df after   engineer_stock_features: ",len(df))
if '3M_return' in df.columns and '3M_return_sector_relative' in df.columns:
    df['3Mreturn_sector_comp']= df.apply(lambda row: 
        safe_subtract(row['3M_return'],row['3M_return_sector_relative']), 
        axis=1)
if '6M_return' in df.columns and '6M_return_sector_relative' in df.columns:
    df['6Mreturn_sector_comp']= df.apply(lambda row: 
        safe_subtract(row['6M_return'],row['6M_return_sector_relative']), 
        axis=1)
    
if '2M_return' in df.columns and '2M_return_sector_relative' in df.columns:
    df['2Mreturn_sector_comp']= df.apply(lambda row: 
        safe_subtract(row['2M_return'],row['2M_return_sector_relative']), 
        axis=1)
if 'peg' in df.columns and 'peg_sector_relative' in df.columns:
    df['peg_sector_comp']= df.apply(lambda row: 
        safe_subtract(row['peg'],row['peg_sector_relative']), 
        axis=1)
df = df.dropna()
print("length of df s: ",len(df))
print()
float_columns = df.select_dtypes(include=['float64', 'float32']).columns
df[float_columns] = df[float_columns].round(2)

# Count the number of rows with to_buy = 0 and to_buy = 1
count_0 = (df['to_buy'] == 0).sum()
count_1 = (df['to_buy'] == 1).sum()
print(f" count 0 : {count_0}")
print(f" count 1 : {count_1}")
rows_to_remove = abs(count_1 - count_0)
if rows_to_remove > 0:
    if count_1>count_0:
        indices_to_remove = df[df['to_buy'] == 1].index
    else :
        indices_to_remove = df[df['to_buy'] == 0].index
    indices_to_remove = random.sample(list(indices_to_remove), rows_to_remove)
    df = df.drop(indices_to_remove)


df.to_csv('metrics/Metric6/ml/processed_data.csv', index=False)