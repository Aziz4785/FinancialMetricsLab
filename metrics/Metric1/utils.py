import pandas as pd
from datetime import datetime, timedelta ,date
import random
from typing import List, Dict,Union
import requests
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time
import pandas as pd
from datetime import datetime, timedelta
import json
import logging
from dotenv import load_dotenv
import os
from requests.exceptions import Timeout ,ConnectionError,RequestException
import pickle

load_dotenv()

# Access the API key
api_key = os.getenv('FMP_API_KEY')



def get_frequency(df):
    # Calculate the difference in days between consecutive dates
    date_diff = df['date'].diff().dt.days

    # Determine frequency based on median differences
    median_diff = date_diff.median()

    if median_diff < 120:
        return 'quarterly'
    elif median_diff < 300:
        return 'semi_annual'
    else:
        return 'yearly'
    
def load_models():
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric1/models/xgboost200_model_8_4.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric1/models/scaler_8_4.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric1/models/rf_200_4_6.pkl', 'rb') as model_file:
        model2 = pickle.load(model_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric1/models/scaler_4_6.pkl', 'rb') as scaler_file:
        scaler2 = pickle.load(scaler_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric1/models/xgboost_model_10_6.pkl', 'rb') as model_file:
        model3 = pickle.load(model_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric1/models/scaler_10_6.pkl', 'rb') as scaler_file:
        scaler3 = pickle.load(scaler_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric1/models/xgboost_model_11_7.pkl', 'rb') as model_file:
        model4 = pickle.load(model_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric1/models/scaler_11_7.pkl', 'rb') as scaler_file:
        scaler4 = pickle.load(scaler_file)
    return model,scaler,model2,scaler2,model3,scaler3,model4,scaler4

def fetch_data_for_ml(stocks):
    sma_10d_dict = {}
    sma_50w_dict = {}
    sma_100d_dict = {}
    sma_200d_dict = {}
    income_dict = {}
    market_cap_dict = {}
    for i,stock in enumerate(stocks):
        sma_10d_dict[stock]=get_SMA(stock, 'all', '1day', 10,None,use_api_call=True)
        sma_50w_dict[stock]=get_SMA(stock, 'all', '1week', 50,None,use_api_call=True)
        sma_100d_dict[stock]=get_SMA(stock, 'all', '1day', 100,None,use_api_call=True)
        sma_200d_dict[stock]=get_SMA(stock, 'all', '1day', 200,None,use_api_call=True)
        income_dict[stock]=get_income_statements(stock, 'quarter')
        market_cap_dict[stock] = get_historical_market_cap('all',stock,sorted_data=None)
        if i%25==0:
            print(f"i = {i} we sleep")
            time.sleep(20)

    return sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict ,market_cap_dict

def load_stocks(nbr,path='sp500_final.csv'):
    stocks_df = pd.read_csv(path)
    stocks = stocks_df['Ticker'].tolist()
    stocks = random.sample(stocks, nbr)
    return stocks

def fetch_stock_data(stocks):
    historical_data_for_stock = {}
    hist_data_df_for_stock = {}
    eps_date_dict = {}
    income_dict = {}
    for c, stock in enumerate(stocks):
        historic_data = get_historical_price(stock)
        date_to_close = convert_to_dict(historic_data)
        historical_data_for_stock[stock] = date_to_close
        hist_data_df_for_stock[stock] = convert_to_df(date_to_close)
        #income_dict[stock]=get_income_statements(stock, 'quarter')
        eps_date_dict[stock]= get_current_and_future_estim_eps(stock, 'all',prefetched_sorted_data=None,income_state=income_dict.get(stock))

        if c % 70 == 0 and c > 0:
            print("we sleep")
            time.sleep(50)

    return historical_data_for_stock, hist_data_df_for_stock,eps_date_dict,income_dict

def convert_to_dict(historic_data):
    if historic_data is None:
        return None
    date_to_close = {}
    for entry in historic_data:
        try:
            date_obj = datetime.strptime(entry['date'], '%Y-%m-%d %H:%M:%S').date()
            date_to_close[date_obj] = entry['close']
        except ValueError as e:
            print(f"Date format error in entry {entry}: {e}")
    return date_to_close

def get_historical_price(symbol):
    url_price = f'https://financialmodelingprep.com/api/v3/historical-chart/1day/{symbol}?from=2018-01-01&to=2024-10-01&apikey={api_key}'
    try:
        response = requests.get(url_price, timeout=2)  # Increased timeout to 2 seconds
        response.raise_for_status()  # Raises an HTTPError for bad responses
    except Timeout:
        print(f"Request timed out for symbol {symbol}")
        return None
    except ConnectionError:
        print(f"Connection error occurred for symbol {symbol}")
        return None
    except RequestException as e:
        print(f"An error occurred while fetching data for symbol {symbol}: {str(e)}")
        return None

    try:
        data_price = response.json()
    except json.JSONDecodeError:
        print(f"Failed to decode JSON for symbol {symbol}. Response content: {response.text[:200]}...")
        return None

    return data_price

def is_quarterly_spaced(data: List[Dict], target_date: Union[str, datetime]) -> bool:
    def ensure_datetime(date_value):
        if isinstance(date_value, str):
            return datetime.strptime(date_value, '%Y-%m-%d')
        elif isinstance(date_value, datetime):
            return date_value
        else:
            raise ValueError(f"Unsupported date type: {type(date_value)}")

    # Convert dates to datetime objects if they're not already
    valid_data = []
    for item in data:
        if item is not None:
            # Create a copy of the item to avoid modifying the original
            item_copy = item.copy()
            item_copy['date'] = ensure_datetime(item['date'])
            valid_data.append(item_copy)
    
    target_date = ensure_datetime(target_date)
    
    # Sort the valid data by date
    sorted_data = sorted(valid_data, key=lambda x: x['date'])
    
    # Find the index of the target date or the closest date before it
    target_index = next((i for i, item in enumerate(sorted_data) if item['date'] >= target_date), len(sorted_data) - 1)
    
    # Get the five elements centered around the target date
    start_index = max(0, target_index - 2)

    end_index = min(len(sorted_data), start_index + 5)

    relevant_data = sorted_data[start_index:end_index]
    #print("relevant data : ")
    #print(relevant_data)
    # Check if we have exactly 5 elements
    if len(relevant_data) != 5:
        return False
    
    # Check if the dates are roughly quarterly spaced
    for i in range(1, len(relevant_data)):
        days_diff = (relevant_data[i]['date'] - relevant_data[i-1]['date']).days
        #print(f" {relevant_data[i]['date']}  -  {relevant_data[i-1]['date']}   -> {days_diff} days")
        if abs(days_diff) > 126:  # Allow 30 days of flexibility
            return False
    
    return True

def generate_valid_dates(start_date, years, exclude_year):
    date_range = pd.date_range(start=start_date - pd.DateOffset(years=years), end=start_date, freq='D')
    valid_dates = [
        date for date in date_range 
        if date.year != exclude_year and date.weekday() not in [5, 6]  # Exclude Saturdays and Sundays
    ]
    return valid_dates

def get_key_metrics(symbol,period):
  url_key_metrics = f'https://financialmodelingprep.com/api/v3/key-metrics/{symbol}?period={period}&limit=300&apikey={api_key}'
  try:
    response_key_metrics = requests.get(url_key_metrics, timeout=1)
  except Timeout:
    print("Request timed out")
    return None
  except ConnectionError:
      print("connection error")
      return None
  if response_key_metrics.status_code != 200:
      print(f"API request failed with status code {response_key_metrics.status_code}: {response_key_metrics.text}")
      return None
  try:
    data_key_metrics = response_key_metrics.json()
  except json.JSONDecodeError:
      print(f"Failed to decode JSON. Response content: {response_key_metrics.text[:200]}...")
      return None
  return data_key_metrics


def extract_eps_from_income_statement(date,sorted_data_income):
    if sorted_data_income is None:
        return None

    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    elif isinstance(date, datetime):
        date = date.date()

    min_date = date - timedelta(days=100)

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date() 
        if min_date <= entry_date <= date and entry['eps'] is not None and entry['eps'] !=0:
            return entry['eps']
        
    return None

def get_current_and_future_estim_eps(symbol, today_date, prefetched_sorted_data = None,income_state = None):
    #maybe it is better to use this api : https://finnhub.io/docs/api/company-eps-estimates
    def ensure_datetime(date_value):
        if isinstance(date_value, str):
            return datetime.strptime(date_value, "%Y-%m-%d")
        elif isinstance(date_value, pd.Timestamp):
            return date_value.to_pydatetime()
        elif isinstance(date_value, date):
            return datetime.combine(date_value, datetime.min.time())
        elif isinstance(date_value, datetime):
            return date_value
        else:
            print(f"Type of today_date: {type(today_date)}, Value: {today_date}")
            raise ValueError(f"Unsupported date type: {type(date_value)}")

    if today_date != 'all':
        today_date = ensure_datetime(today_date)


    if prefetched_sorted_data is None :
        url = f'https://financialmodelingprep.com/api/v3/analyst-estimates/{symbol}?period=quarter&apikey={api_key}'
        try:
            response_key_metrics = requests.get(url, timeout=1)
        except Timeout:
            print("Request timed out")
            return None,None

        data_eps = response_key_metrics.json()
        
        

        sorted_data = sorted(data_eps, key=lambda x: datetime.strptime(x['date'], "%Y-%m-%d"), reverse=True)
    else:
        sorted_data = prefetched_sorted_data

    if today_date == 'all':
        return sorted_data
    
    if sorted_data is None:
        return None,None
    
    #if not is_quarterly_spaced(sorted_data, today_date):
        #print(f" -> eps are None because they are not quartely spaced for {symbol}, {today_date}")
        #return None,None
    
    current_estimated_eps = None
    future_estimated_eps = None
    #print(sorted_data[:6])
    for i, entry in enumerate(sorted_data):
        if entry is not None:
            entry_date = ensure_datetime(entry['date'])
            #print(f"entry date : {entry_date} type : ({type(entry_date)})")
            #print(f"today_date : {today_date} type : ({type(today_date)})")
            if entry_date <= today_date:
                if current_estimated_eps is None:
                    current_estimated_eps = entry['estimatedEpsAvg']
                
                if i > 0:
                    future_estimated_eps = sorted_data[i-1]['estimatedEpsAvg']
                
                break
    #current_estimated_eps = extract_eps_from_income_statement(today_date,income_state)
    return current_estimated_eps, future_estimated_eps

def get_current_and_future_estim_eps_old(symbol, today_date, prefetched_sorted_data = None):
    def ensure_datetime(date_value):
        if isinstance(date_value, str):
            return datetime.strptime(date_value, "%Y-%m-%d")
        elif isinstance(date_value, pd.Timestamp):
            return date_value.to_pydatetime()
        elif isinstance(date_value, date):
            return datetime.combine(date_value, datetime.min.time())
        elif isinstance(date_value, datetime):
            return date_value
        else:
            raise ValueError(f"Unsupported date type: {type(date_value)}")

    if today_date != 'all':
        today_date = ensure_datetime(today_date)


    if prefetched_sorted_data is None :
        url = f'https://financialmodelingprep.com/api/v3/historical/earning_calendar/{symbol}?apikey={api_key}'
        try:
            response_key_metrics = requests.get(url, timeout=1)
        except Timeout:
            print("Request timed out")
            return None,None

        data_eps = response_key_metrics.json()
        
        

        sorted_data = sorted(data_eps, key=lambda x: datetime.strptime(x['date'], "%Y-%m-%d"), reverse=True)
    else:
        sorted_data = prefetched_sorted_data

    if today_date == 'all':
        return sorted_data
    
    if sorted_data is None:
        return None
    
    if not is_quarterly_spaced(sorted_data, today_date):
        #print(f" -> eps are None because they are not quartely spaced for {symbol}, {today_date}")
        return None,None
    
    current_eps = None
    estimated_eps = None
    #print(sorted_data[:6])
    for i, entry in enumerate(sorted_data):
        if entry is not None:
            entry_date = ensure_datetime(entry['date'])
            #print(f"entry date : {entry_date} type : ({type(entry_date)})")
            #print(f"today_date : {today_date} type : ({type(today_date)})")
            if entry_date <= today_date:
                if current_eps is None:
                    current_eps = entry['eps']
                
                if i > 0:
                    estimated_eps = sorted_data[i-1]['epsEstimated']
                
                break
    
    return current_eps, estimated_eps

def calculate_quarter_peg_ratio(symbol,date,price_at_date,prefetched_data=None,incomes=None):
    current_quarter_eps,future_quarter_eps = get_current_and_future_estim_eps(symbol, date,prefetched_data,incomes)
    if current_quarter_eps is None or future_quarter_eps is None or current_quarter_eps ==0:
       return None
    Earnings_growth_rate = (future_quarter_eps-current_quarter_eps)/current_quarter_eps
    if  Earnings_growth_rate is None or Earnings_growth_rate<=0  or price_at_date is None or Earnings_growth_rate==0:
      return None
    else:
       peg = (price_at_date/ current_quarter_eps) / (Earnings_growth_rate*100)
       if peg<0 or peg>300:
           return None
       return peg
    
def calculate_peg_ratio_old(symbol,date,price_at_date,eps_calculation,data_key_metrics=None,debug=False):
  period = 'quarter'
  if data_key_metrics is None:
    if period != 'quarter' and eps_calculation =='quarter':
      print("warning incoherency in calculate_peg_ratio")
    url_key_metrics = f'https://financialmodelingprep.com/api/v3/key-metrics/{symbol}?period={period}&limit=300&apikey={api_key}'
    try:
      response_key_metrics = requests.get(url_key_metrics, timeout=1)
    except Timeout:
      print("Request timed out")
      return None
    except ConnectionError:
        print("ConnectionError")
        return None
        
    data_key_metrics = response_key_metrics.json()
  if isinstance(date, str):
        date = pd.to_datetime(date)
  # Extract PE ratios
  pe_ratios = [item.get('peRatio') for item in data_key_metrics][::-1]
  pe_ratios = [item.get('peRatio') for item in data_key_metrics][::-1]
  dates = [item.get('date') for item in data_key_metrics][::-1]
  eps = [item.get('netIncomePerShare') for item in data_key_metrics][::-1]
  df = pd.DataFrame({'date': dates, 'pe_ratios': pe_ratios, 'eps':eps})
  df['date'] = pd.to_datetime(df['date'])
  df = df[df['date'] >= '2018-01-01']
  if debug :
    print("df : ")
    print(df)
  date_after_1Y = None
  if eps_calculation =='ttm':
    date_after_1Y = date + pd.DateOffset(years=1)
  elif eps_calculation =='quarter':
    date_after_1Y = date + pd.DateOffset(months=3)
  input_date = pd.to_datetime(date)

  # Determine the frequency of the DataFrame
  frequency = get_frequency(df)
  if frequency != 'quarterly' and eps_calculation =='quarter':
      return None
  # Filter rows with date less than or equal to input_date
  filtered_df = df[df['date'] <= input_date]
  future_df = df[df['date'] <= date_after_1Y]
  # Calculate TTM EPS based on frequency
  ttm_eps = None
  ttm_eps_future = None
  if frequency == 'quarterly':
      # Sum the last 4 EPS values
    tail_length=4
    if eps_calculation == 'ttm':
        tail_length=4
    elif eps_calculation == 'quarter':
        tail_length=1
    if len(filtered_df) >= tail_length:
        ttm_eps = filtered_df['eps'].tail(tail_length).sum()
    if len(future_df) >= tail_length:
        ttm_eps_future = future_df['eps'].tail(tail_length).sum()
  elif frequency == 'semi_annual':
      # Sum the last 2 EPS values
      if len(filtered_df) >= 2:
        ttm_eps = filtered_df['eps'].tail(2).sum()
      if len(future_df) >= 2:
        ttm_eps_future = future_df['eps'].tail(2).sum()
  elif frequency == 'yearly':
      # Use the last EPS value available
      if len(filtered_df) >= 1:
        ttm_eps = filtered_df['eps'].iloc[-1] if not filtered_df.empty else None
      if len(future_df) >= 1:
        ttm_eps_future = future_df['eps'].iloc[-1] if not future_df.empty else None
  if debug :
    print(f" price_at_date = {price_at_date}, ttm_eps = {ttm_eps} , ttm_eps_future = {ttm_eps_future}")
  peg = None
  if price_at_date is not None and ttm_eps is not None and ttm_eps_future is not None and ttm_eps>0 and ttm_eps_future>0:
    Earnings_growth_rate = (ttm_eps_future-ttm_eps)/ttm_eps
    if Earnings_growth_rate<=0:
      return None
    if debug :
        print("pe = ",price_at_date/ ttm_eps)
    peg = (price_at_date/ ttm_eps) / (Earnings_growth_rate*100)
  return peg



def extract_costOfRevenue(date,sorted_data_income):
    if sorted_data_income is None:
        return None

    min_date = date - timedelta(days=370)

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d")
        if min_date <= entry_date <= date and entry['costOfRevenue'] is not None:
            return entry['costOfRevenue']
        
    return None

def extract_netIncome(date,sorted_data_income):
    if sorted_data_income is None:
        return None

    min_date = date - timedelta(days=370)

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d")
        if min_date <= entry_date <= date and entry['netIncome'] is not None:
            return entry['netIncome']
        
    return None

def get_max_price_in_range(symbol, start_date, end_date,historical_data=None):
    # Convert dates to datetime objects if they're strings
    if isinstance(start_date, str):
        start_date = datetime.strptime(start_date, '%Y-%m-%d')
    elif isinstance(start_date, pd.Timestamp):
        start_date = start_date.date()
    if isinstance(end_date, str):
        end_date = datetime.strptime(end_date, '%Y-%m-%d')
    elif isinstance(end_date, pd.Timestamp):
        end_date = end_date.date()
    start_date = start_date + timedelta(days=1)
    if historical_data is not None:
        start_datetime = pd.to_datetime(start_date)
        end_datetime = pd.to_datetime(end_date)
        max_price = 0
        # for key,value in historical_data.items():
        #     if start_date <= key <= end_date:
        #         if value > max_price:
        #             max_price = value
        mask = (historical_data.index >= start_datetime) & (historical_data.index <= end_datetime)
        relevant_data = historical_data.loc[mask, 'close']

        if relevant_data.empty:
            return None
        max_price = relevant_data.max()
        return max_price

    # Format dates for API request
    start_date_str = start_date.strftime('%Y-%m-%d')
    end_date_str = end_date.strftime('%Y-%m-%d')

    # Get historical stock data using the financial modeling API
    url_price = f'https://financialmodelingprep.com/api/v3/historical-chart/1day/{symbol}?from={start_date_str}&to={end_date_str}&apikey={api_key}'
    try:
        response = requests.get(url_price, timeout=1)
    except Timeout:
        print("Request timed out")
        return None
    data_price = response.json()
    # Validate the data received
    if not data_price or len(data_price) == 0:
        return None

    # Find the maximum price in the range
    max_price = max([entry['close'] for entry in data_price if 'close' in entry])

    return max_price


def generate_valid_random_date():
    today = datetime.now()
    while True:
        random_date = pd.Timestamp(today - pd.DateOffset(years=5)) + pd.to_timedelta(np.random.randint(0, (365*4 + 1)), unit='D')
        random_date = random_date.floor('D')
        
        # Skip dates in 2020
        if random_date.year == 2020:
            continue
        
        # Adjust weekend dates
        if random_date.dayofweek == 5:  # Saturday
            random_date -= pd.Timedelta(days=1)  # Move to Friday
        elif random_date.dayofweek == 6:  # Sunday
            random_date += pd.Timedelta(days=1)  # Move to Monday
        
        return random_date
    
def convert_to_df(date_to_close):
    if date_to_close is None:
       return None
    date_to_close_df = pd.DataFrame(date_to_close.items(), columns=['date', 'close'])
    date_to_close_df['date'] = pd.to_datetime(date_to_close_df['date'])
    date_to_close_df.set_index('date', inplace=True)
    date_to_close_df.sort_index(inplace=True)
    return date_to_close_df

def get_income_statements(symbol,period):
    url_income = f'https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period={period}&apikey={api_key}'

    try:
        response_income = requests.get(url_income, timeout=3)
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None

    if response_income.status_code != 200:
        print(f"Error: API returned status code {response_income.status_code}")
        return None
    data_income = response_income.json()
    sorted_data = sorted(data_income, key=lambda x: datetime.strptime(x['date'].split()[0], "%Y-%m-%d"), reverse=True)

    return sorted_data
"""
def extract_revenue(input_date,sorted_data_income):
    if sorted_data_income is None:
        return None

    if isinstance(input_date, date):
        input_date = datetime.combine(input_date, datetime.min.time())
    
    
    min_date = input_date - timedelta(days=100)

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d")
        if min_date <= entry_date <= input_date and entry['revenue'] is not None and entry['revenue'] !=0:
            return entry['revenue']
        
    return None
"""
def extract_revenue(input_date, sorted_data_income):
    if sorted_data_income is None:
        return None, None

    if isinstance(input_date, date):
        input_date = datetime.combine(input_date, datetime.min.time())
    if isinstance(input_date, str):
        input_date = datetime.strptime(input_date, '%Y-%m-%d').date()
    elif isinstance(input_date, pd.Timestamp):
        input_date = input_date.date()
    elif isinstance(input_date, datetime):
        input_date = input_date.date()
        
    min_date = input_date - timedelta(days=100)
    
    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if min_date <= entry_date <= input_date and entry['revenue'] is not None and entry['revenue'] != 0:
            lag_days = (input_date - entry_date).days
            return entry['revenue'], lag_days
        
    return None, None

def predict_buy(model,scaler,date,symbol,peg_ratio,features_for_pred,price,sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict ,market_cap_dict):
    if income_dict is not None and not isinstance(income_dict, dict):
        ebitdaMargin = extract_ebitda_margin(date, income_dict)
    elif isinstance(income_dict, dict):
        ebitdaMargin = extract_ebitda_margin(date, income_dict[symbol])

    feature_calculations = {
        'MarkRevRatio': lambda: calculate_mark_rev_ratio(date, symbol, market_cap_dict, income_dict),
        'sma_50w': lambda: get_SMA(symbol, date, '1week', 50, sorted_data=None if sma_50w_dict is None else sma_50w_dict[symbol],use_api_call=True),
        'sma_200d': lambda: get_SMA(symbol, date, '1day', 200, sorted_data=None if sma_200d_dict is None else sma_200d_dict[symbol],use_api_call=True),
        'sma_100d': lambda: get_SMA(symbol, date, '1day', 100, sorted_data=None if sma_100d_dict is None else sma_100d_dict[symbol],use_api_call=True),
        'sma_10d': lambda: get_SMA(symbol, date, '1day', 10, sorted_data=None if sma_10d_dict is None else sma_10d_dict[symbol],use_api_call=True),
        'ebitdaMargin': lambda: ebitdaMargin,
        'price': lambda: price,
        'ratio': lambda: peg_ratio
    }

    # Calculate only the required features
    data_dict = {'date': date, 'symbol': symbol}
    for feature in features_for_pred:
        if feature in feature_calculations:
            data_dict[feature] = feature_calculations[feature]()

    
    data = pd.DataFrame([data_dict])

    # Convert date to datetime if it's not already
    data['date'] = pd.to_datetime(data['date'])

    if any(data_dict.get(feature) is None for feature in features_for_pred):
        return None, None
    
    X = data[features_for_pred]
    X_scaled = scaler.transform(X)

    prediction = model.predict(X_scaled)
    #probability = loaded_model.predict_proba(X_scaled)[0, 1]  # Probability of class 1 (buy)
    probability=1
    return prediction[0], probability


def calculate_mark_rev_ratio(date, symbol, market_cap_dict, income_dict):
    if market_cap_dict is not None and not isinstance(market_cap_dict, dict):
        market_cap = market_cap_dict
    else:
        market_cap = get_historical_market_cap(date, symbol, sorted_data=None if market_cap_dict is None else market_cap_dict[symbol])
    if income_dict is None:
        print(" warning : income dict is None")
        revenue,_ = extract_revenue(date, get_income_statements(symbol, 'quarter')) 
    else: 
        if not isinstance(income_dict, dict):
             revenue,_ = extract_revenue(date, income_dict)
        elif isinstance(income_dict, dict):
             revenue,_ = extract_revenue(date, income_dict[symbol])
       
    
    if market_cap is not None and revenue is not None and revenue != 0:
        return market_cap / revenue
    else:
        return None
    

    
def extract_interestCoverage(date,sorted_data_income):
    if sorted_data_income is None:
        return None

    min_date = date - timedelta(days=370)

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d")
        if min_date <= entry_date <= date and entry['operatingIncome'] is not None and entry['interestExpense'] is not None and entry['interestExpense'] !=0:
            return entry['operatingIncome']/entry['interestExpense']
        
    return None

def extract_ebitda_margin(date,sorted_data_income):
    if sorted_data_income is None:
        return None
    
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    elif isinstance(date, datetime):
        date = date.date()

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if entry_date <= date and entry['revenue'] is not None and entry['ebitda'] is not None and entry['revenue'] !=0:
            return entry['ebitda']/entry['revenue']
        
    return None

def extract_net_profit_margin(date,sorted_data_income):
    if sorted_data_income is None:
        return None

    min_date = date - timedelta(days=370)

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d")
        if min_date <= entry_date <= date and entry['revenue'] is not None and entry['netIncome'] is not None and entry['revenue'] !=0:
            return entry['netIncome']/entry['revenue']
        
    return None
def extract_other_expenses(date,sorted_data_income):
    if sorted_data_income is None:
        return None

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d")
        if entry_date <= date:
            return entry['otherExpenses']
    return None

def get_historical_market_cap(date,stock,sorted_data=None):
    if date != 'all':
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d').date()
        elif isinstance(date, pd.Timestamp):
            date = date.date()
        elif isinstance(date, datetime):
            date = date.date()
    if sorted_data==None:
        url_market_cap = f'https://financialmodelingprep.com/api/v3/historical-market-capitalization/{stock}?from=2018-10-10&to=2024-10-20&apikey={api_key}'

        try:
            response_sma = requests.get(url_market_cap, timeout=1)
        except requests.exceptions.Timeout:
            print("Request timed out")
            return None

        if response_sma.status_code != 200:
            print(f"Error: API returned status code {response_sma.status_code}")
            return None
        data_sma = response_sma.json()
        sorted_data = sorted(data_sma, key=lambda x: datetime.strptime(x['date'].split()[0], "%Y-%m-%d"), reverse=True)
    if date=='all':
        return sorted_data
    min_date = date - timedelta(days=5)

    for entry in sorted_data:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if min_date<=entry_date <= date:
            return entry['marketCap']
    
    return None

def get_SMA(symbol, date, period,length,sorted_data=None,use_api_call=False):
    if date != 'all':
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d').date()
        elif isinstance(date, pd.Timestamp):
            date = date.date()
        elif isinstance(date, datetime):
            date = date.date()

    if sorted_data==None and use_api_call==True:
        url_sma = f'https://financialmodelingprep.com/api/v3/technical_indicator/{period}/{symbol}?type=sma&period={length}&apikey={api_key}'
        try:
            response_sma = requests.get(url_sma, timeout=3)
        except requests.exceptions.Timeout:
            print("Request timed out")
            return None

        if response_sma.status_code != 200:
            print(f"Error: API returned status code {response_sma.status_code}")
            return None
        data_sma = response_sma.json()
        sorted_data = sorted(data_sma, key=lambda x: datetime.strptime(x['date'].split()[0], "%Y-%m-%d"), reverse=True)

    if date=='all':
        return sorted_data

    if sorted_data is not None:
        for entry in sorted_data:
            entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
            if entry_date <= date:
                return entry['sma']
    
    return None

def get_stock_price_at_date(symbol, date,historical_data =None,not_None=False):
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    start_date_str = date.strftime('%Y-%m-%d')
    end_date = date + timedelta(days=1)
    end_date_str = end_date.strftime('%Y-%m-%d')

    if historical_data is not None:
        if not_None==True:
           #if Nnot_None is True i want to get the most recent date before date
           for days_back in range(4): 
                test_date = date - timedelta(days=days_back)
                if test_date in historical_data:
                    return historical_data[test_date]
           return None  # If no data found in the last 4 days
        return historical_data.get(date)
    else:
       return None
       print("historical_data is None for stock ",symbol)
    url_price = f'https://financialmodelingprep.com/api/v3/historical-chart/1day/{symbol}?from={start_date_str}&to={end_date_str}&apikey={api_key}'
    #print("waiting for response....")
    try:
        response = requests.get(url_price, timeout=1)
    except Timeout:
        print("Request timed out")
        return None
   # print("response received")
    #print(response)
    if response.status_code != 200:
        print(f"API request failed with status code {response.status_code}: {response.text}")
        return None
    try:
        data_price = response.json()
    except json.JSONDecodeError:
        print(f"Failed to decode JSON. Response content: {response.text[:200]}...")
        raise
    if not data_price or len(data_price) == 0:
        return None
    for entry in data_price:
        if 'date' in entry:
            entry_date = datetime.strptime(entry['date'], '%Y-%m-%d %H:%M:%S').date()
            if entry_date == date:
                return entry['close']
    return None

def get_current_price(symbol):
    url = f'https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={api_key}'
    try:
      response_key_metrics = requests.get(url, timeout=2)
    except Timeout:
      print("Request timed out")
      return None
    data_price = response_key_metrics.json()
    current_price = data_price[0].get('price')
    return current_price

def calculate_current_quarter_peg(symbol,current_price):
    today_date = pd.Timestamp.today()
    url_key_metrics = f'https://financialmodelingprep.com/api/v3/key-metrics/{symbol}?period=quarter&limit=4&apikey={api_key}'
    try:
      response_key_metrics = requests.get(url_key_metrics, timeout=1)
    except Timeout:
      print("Request timed out")
      return None
    data_key_metrics = response_key_metrics.json()
    pe = data_key_metrics[0].get('peRatio')
    current_quarter_eps,future_quarter_eps = get_current_and_future_estim_eps(symbol, today_date)
    if current_quarter_eps is None or future_quarter_eps is None:
       return None
    Earnings_growth_rate = (future_quarter_eps-current_quarter_eps)/current_quarter_eps
    if Earnings_growth_rate<=0 or Earnings_growth_rate is None:
      return None
    else:
       peg = (current_price/ current_quarter_eps) / (Earnings_growth_rate*100)
       return peg

def get_current_market_cap(stock):
  url = f'https://financialmodelingprep.com/api/v3/market-capitalization/{stock}?apikey={api_key}'
  try:
      response_sma = requests.get(url, timeout=2)
  except requests.exceptions.Timeout:
      print("Request timed out")
      return None
  if response_sma.status_code != 200:
      print(f"Error: API returned status code {response_sma.status_code}")
      return None
  data = response_sma.json()
  if len(data)>0:
    return data[0]['marketCap']
  else:
    return None