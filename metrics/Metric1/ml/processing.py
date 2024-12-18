import pandas as pd
import yfinance as yf
import random
from datetime import datetime
import sys
import os
from ..utils import *



"""
to run : to run : py -m metrics.Metric1.ml.processing
if there is a lot of None values at the end, re-run this script because sometimes the problem comes from the API
"""
df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric1/ml/cleaned_training_data.csv', parse_dates=['date'])

stocks = df['symbol'].unique()
print(f"number of unique stocks : {len(stocks)}")
if (len(stocks) <30):
    print("too few stocks ... cannot perform analysis on it")
    exit()


sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict ,market_cap_dict = fetch_data_for_ml(stocks)

df['sma_10d'] = df.apply(lambda row: get_SMA(row['symbol'], row['date'], '1day', 10, sma_10d_dict.get(row['symbol']),use_api_call=False), axis=1)
df['sma_50w'] = df.apply(lambda row: get_SMA(row['symbol'], row['date'], '1week', 50, sma_50w_dict.get(row['symbol']),use_api_call=False), axis=1)
df['sma_100d'] = df.apply(lambda row: get_SMA(row['symbol'], row['date'], '1day', 100, sma_100d_dict.get(row['symbol']),use_api_call=False), axis=1)
df['sma_200d'] = df.apply(lambda row: get_SMA(row['symbol'], row['date'], '1day', 200, sma_200d_dict.get(row['symbol']),use_api_call=False), axis=1)

print("--------")
print("length of df: ",len(df))
# Add other_expenses column
df['other_expenses'] = df.apply(lambda row: extract_other_expenses(row['date'], income_dict[row['symbol']]), axis=1)
df['ebitdaMargin'] = df.apply(lambda row: extract_ebitda_margin(row['date'], income_dict[row['symbol']]), axis=1)
df['eps'] = df.apply(lambda row: extract_eps_from_income_statement(row['date'], income_dict[row['symbol']]), axis=1)
df['costOfRevenue'] = df.apply(lambda row: extract_costOfRevenue(row['date'], income_dict[row['symbol']]), axis=1)
df['revenues'], df['revenues_lag_days'] = zip(*df.apply(
    lambda row: extract_revenue(row['date'], income_dict[row['symbol']]), 
    axis=1
))
df['MarkRevRatio'] = df.apply(lambda row: calculate_mark_rev_ratio(row['date'],row['symbol'], market_cap_dict, income_dict), axis=1)
print("-------")
print("length of df: ",len(df))
none_counts = {
    'ebitdaMargin': df['ebitdaMargin'].isnull().sum(),
    'costOfRevenue': df['costOfRevenue'].isnull().sum(),
    'eps': df['eps'].isnull().sum(),
    'MarkRevRatio': df['MarkRevRatio'].isnull().sum()
}
# Print the results
for column, count in none_counts.items():
    print(f"Number of None values in {column}: {count}")

df = df.dropna()
print("-------")
print("length of df: ",len(df))
df['price'] = df['price'].round(2)
df['ratio'] = df['ratio'].round(2)
df['sma_10d'] = df['sma_10d'].round(2)
df['sma_50w'] = df['sma_50w'].round(2)
df['sma_100d'] = df['sma_100d'].round(2)
df['sma_200d'] = df['sma_200d'].round(2)
df['eps'] = df['eps'].round(2)
df['costOfRevenue'] = df['costOfRevenue'].round(2)
df['ebitdaMargin'] = df['ebitdaMargin'].round(3)
df['revenues'] = df['revenues'].round(2)
df['MarkRevRatio'] = df['MarkRevRatio'].round(3)
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
