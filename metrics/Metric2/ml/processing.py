import pandas as pd
import yfinance as yf
import random
from datetime import datetime
import sys
import os
from ..utils import *



"""
to run : to run : py -m metrics.Metric2.ml.processing
if there is a lot of None values at the end, re-run this script because sometimes the problem comes from the API
"""
df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric2/ml/cleaned_training_data.csv', parse_dates=['date'])

stocks = df['symbol'].unique()
print(f"number of unique stocks : {len(stocks)}")
if (len(stocks) <30):
    print("too few stocks ... cannot perform analysis on it")
    exit()


sma_10d_dict,sma_50w_dict,sma_100d_dict,sma_200d_dict,sma_50d_dict,income_dict = fetch_data_for_ml(stocks)

df['sma_10d'] = df.apply(lambda row: get_SMA(row['symbol'], row['date'], '1day', 10, sma_10d_dict[row['symbol']]), axis=1)
print("SMA 10D done !")
df['sma_50w'] = df.apply(lambda row: get_SMA(row['symbol'], row['date'], '1week', 50, sma_50w_dict[row['symbol']]), axis=1)
print("SMA 50W done !")
df['sma_100d'] = df.apply(lambda row: get_SMA(row['symbol'], row['date'], '1day', 100, sma_100d_dict[row['symbol']]), axis=1)
df['sma_200d'] = df.apply(lambda row: get_SMA(row['symbol'], row['date'], '1day', 200, sma_200d_dict[row['symbol']]), axis=1)
df['sma_50d'] = df.apply(lambda row: get_SMA(row['symbol'], row['date'], '1day', 50, sma_50d_dict[row['symbol']]), axis=1)

# Add other_expenses column
df['other_expenses'] = df.apply(lambda row: extract_other_expenses(row['date'], income_dict[row['symbol']]), axis=1)
df['NetProfitMargin'] = df.apply(lambda row: extract_net_profit_margin(row['date'], income_dict[row['symbol']]), axis=1)
df['ebitdaMargin'] = df.apply(lambda row: extract_ebitda_margin(row['date'], income_dict[row['symbol']]), axis=1)
df['eps'] = df.apply(lambda row: extract_eps(row['date'], income_dict[row['symbol']]), axis=1)
df['costOfRevenue'] = df.apply(lambda row: extract_costOfRevenue(row['date'], income_dict[row['symbol']]), axis=1)
df['netIncome'] = df.apply(lambda row: extract_netIncome(row['date'], income_dict[row['symbol']]), axis=1)
df['revenues'], df['revenues_lag_days'] = zip(*df.apply(
    lambda row: extract_revenue(row['date'], income_dict[row['symbol']]), 
    axis=1
))
df = add_pe_ratio(df)

df['price'] = df['price'].round(2)
df['ratio'] = df['ratio'].round(2)
df['sma_10d'] = df['sma_10d'].round(2)
df['sma_50w'] = df['sma_50w'].round(2)
df['sma_50d'] = df['sma_50d'].round(2)
df['sma_100d'] = df['sma_100d'].round(2)
df['sma_200d'] = df['sma_200d'].round(2)
df['eps'] = df['eps'].round(2)
df['costOfRevenue'] = df['costOfRevenue'].round(2)
df['netIncome'] = df['netIncome'].round(2)
df['NetProfitMargin'] = df['NetProfitMargin'].round(3)
df['ebitdaMargin'] = df['ebitdaMargin'].round(3)
df['revenues'] = df['revenues'].round(2)

none_counts = {
    'NetProfitMargin': df['NetProfitMargin'].isnull().sum(),
    'ebitdaMargin': df['ebitdaMargin'].isnull().sum(),
    'netIncome': df['netIncome'].isnull().sum(),
    'costOfRevenue': df['costOfRevenue'].isnull().sum(),
    'eps': df['eps'].isnull().sum(),
}

# Print the results
for column, count in none_counts.items():
    print(f"Number of None values in {column}: {count}")
    
df = df.dropna()
df['cost_revenue_ratio'] = df.apply(lambda x: x['costOfRevenue'] / x['revenues'] if x['revenues'] != 0 else None, axis=1)
df['cost_revenue_ratio'] = df['cost_revenue_ratio'].round(2)

# Count the number of rows with to_buy = 0 and to_buy = 1
count_0 = (df['to_buy'] == 0).sum()
count_1 = (df['to_buy'] == 1).sum()
print(f" count 0 : {count_0}")
print(f" count 1 : {count_1}")
rows_to_remove = count_1 - count_0
if rows_to_remove > 0:
    indices_to_remove = df[df['to_buy'] == 1].index
    indices_to_remove = random.sample(list(indices_to_remove), rows_to_remove)
    df = df.drop(indices_to_remove)


df.to_csv('processed_data.csv', index=False)
