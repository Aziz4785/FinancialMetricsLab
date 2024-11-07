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

def load_stocks(nbr,path='sp500_final.csv'):
    stocks_df = pd.read_csv(path)
    stocks = stocks_df['Ticker'].tolist()
    stocks = random.sample(stocks, nbr)
    return stocks

def fetch_stock_data(stocks):
    historical_data_for_stock = {}
    market_cap_dict = {}
    revenues_dict = {}
    hist_data_df_for_stock = {}
    for c, stock in enumerate(stocks):
        historic_data = get_historical_price(stock)
        date_to_close = convert_to_dict(historic_data)
        hist_data_df_for_stock[stock] = convert_to_df(date_to_close)
        historical_data_for_stock[stock] = date_to_close

        market_cap_dict[stock] = get_historical_market_cap('all',stock,sorted_data=None)
        revenues_dict[stock] = get_revenues(stock,'quarter') #should be named get_income_statementss

        if c % int(-0.2*len(stocks)+140) == 0 and c > 0:
            print("we sleep")
            time.sleep(50)

    return historical_data_for_stock, market_cap_dict, revenues_dict, hist_data_df_for_stock

def load_models():
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric2/models/xgboost_model_10_7.pkl', 'rb') as model_file:
        model2 = pickle.load(model_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric2/models/scaler_10_7.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric2/models/rf_200_10_7.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    return model,scaler,model2

def get_income_statements(symbol,period):
    url_income = f'https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period={period}&apikey={api_key}'

    try:
        response_income = requests.get(url_income, timeout=2)
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None

    if response_income.status_code != 200:
        print(f"Error: API returned status code {response_income.status_code}")
        return None
    data_income = response_income.json()
    sorted_data = sorted(data_income, key=lambda x: datetime.strptime(x['date'].split()[0], "%Y-%m-%d"), reverse=True)

    return sorted_data

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


def convert_to_df(date_to_close):
    if date_to_close is None:
       return None
    date_to_close_df = pd.DataFrame(date_to_close.items(), columns=['date', 'close'])
    date_to_close_df['date'] = pd.to_datetime(date_to_close_df['date'])
    date_to_close_df.set_index('date', inplace=True)
    date_to_close_df.sort_index(inplace=True)
    return date_to_close_df


def get_revenues(symbol,period):
    url_income = f'https://financialmodelingprep.com/api/v3/income-statement/{symbol}?period={period}&apikey={api_key}'

    try:
        response_income = requests.get(url_income, timeout=2)
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None

    if response_income.status_code != 200:
        print(f"Error: API returned status code {response_income.status_code}")
        return None
    data_income = response_income.json()
    sorted_data = sorted(data_income, key=lambda x: datetime.strptime(x['date'].split()[0], "%Y-%m-%d"), reverse=True)

    return sorted_data

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
    url_price = f'https://financialmodelingprep.com/api/v3/historical-chart/1day/{symbol}?from=2018-01-01&to=2024-10-16&apikey={api_key}'
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

def add_pe_ratio(df):
    """
    Add PE ratio column to a dataframe containing price and eps columns.
    PE ratio is calculated as price/eps where eps is non-zero and both values are not None.
    
    Parameters:
    df (pandas.DataFrame): DataFrame with 'price' and 'eps' columns
    
    Returns:
    pandas.DataFrame: DataFrame with additional 'pe_ratio' column
    """
    # Create a copy to avoid modifying the original DataFrame
    result = df.copy()
    
    # Calculate PE ratio with conditions
    result['pe_ratio'] = None  # Initialize with None
    
    # Create mask for valid calculations
    valid_mask = (
        result['eps'].notna() &  # eps is not None
        result['price'].notna() &  # price is not None
        (result['eps'] != 0)  # eps is not zero
    )
    
    # Calculate PE ratio only where conditions are met
    result.loc[valid_mask, 'pe_ratio'] = (
        result.loc[valid_mask, 'price'] / result.loc[valid_mask, 'eps']
    )
    
    return result

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


def fetch_data_for_ml(stocks):
    sma_10d_dict = {}
    sma_50w_dict = {}
    sma_100d_dict = {}
    sma_200d_dict = {}
    sma_50d_dict = {}
    income_dict = {}
    for i,stock in enumerate(stocks):
        sma_10d_dict[stock]=get_SMA(stock, 'all', '1day', 10)
        #sma_50w_dict[stock]=get_SMA(stock, 'all', '1week', 50)
        sma_100d_dict[stock]=get_SMA(stock, 'all', '1day', 100)
        sma_200d_dict[stock]=get_SMA(stock, 'all', '1day', 200)
        #sma_50d_dict[stock]=get_SMA(stock, 'all', '1day', 50)
        income_dict[stock]=get_income_statements(stock, 'quarter')
        if i%20==0:
            print(f"i = {i} we sleep")
            time.sleep(20)
    return sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,income_dict

def predict_buy(model,scaler,date,symbol,features_for_pred,price,pr_ratio,sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict):
    # ['revenues', 'price', 'pe_ratio', 'ebitdaMargin', 'ratio', 'sma_100d', 'sma_200d']
    if income_dict is not None and not isinstance(income_dict, dict):
        eps = extract_eps(date, income_dict)
        ebitdaMargin = extract_ebitda_margin(date, income_dict)
        revenues,_= extract_revenue(date, income_dict)
    elif isinstance(income_dict, dict):
        eps = extract_eps(date, income_dict[symbol])
        ebitdaMargin = extract_ebitda_margin(date, income_dict[symbol])
        revenues,_= extract_revenue(date, income_dict[symbol])
    #print("eps : ",eps)
    
    #print("ebitdaMargin : ",ebitdaMargin)
    
    #print("revenues : ",revenues)
    feature_calculations = {
        'sma_50w': lambda: get_SMA(symbol, date, '1week', 50, sorted_data=None if sma_50w_dict is None else sma_50w_dict[symbol]),
        'sma_200d': lambda: get_SMA(symbol, date, '1day', 200, sorted_data=None if sma_200d_dict is None else sma_200d_dict[symbol]),
        'sma_100d': lambda: get_SMA(symbol, date, '1day', 100, sorted_data=None if sma_100d_dict is None else sma_100d_dict[symbol]),
        'sma_10d': lambda: get_SMA(symbol, date, '1day', 10, sorted_data=None if sma_10d_dict is None else sma_10d_dict[symbol]),
        'price': lambda: price,
        'ebitdaMargin': lambda: ebitdaMargin,
        'revenues': lambda: revenues,
        'ratio': lambda: pr_ratio,
        'eps': lambda: eps,
        'pe_ratio': lambda: price / eps if price is not None and eps is not None and eps != 0 else None
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

def calculate_historical_price_to_revenue_ratio(stock,date,market_cap_values,income_statement_values,do_api_call=True):
    if do_api_call==False and (market_cap_values is None or income_statement_values is None):
        return None
    market_cap = get_historical_market_cap(date,stock,sorted_data=market_cap_values,do_api_call=do_api_call)
    revenues,_ = extract_revenue(date,income_statement_values)

    if market_cap is None or revenues is None or revenues<=0:
        return None 
    
    pr_ratio =  market_cap/revenues
    if pr_ratio>=200:
        return None 
    return pr_ratio

def calculate_current_price_to_revenue_ratio(stock,date,market_cap_values,income_statement_values,do_api_call=True):
    if do_api_call==False and (market_cap_values is None or income_statement_values is None):
        return None
    market_cap = get_current_market_cap(stock)
    if income_statement_values is None and do_api_call == True:
        income_statement_values= get_income_statements(stock, 'quarter')
    elif income_statement_values is None and do_api_call == False:
        return None

    revenues,_ = extract_revenue(date, income_statement_values) 

    if market_cap is None or revenues is None or revenues<=0:
        # if market_cap is None:
        #     print(" pr is None because market cap is None")
        # if revenues is None :
        #     print("  pr is None because revenues is NOnE")
        # elif revenues<=0:
        #     print(" pr is None because revenues <=0")
        return None 
    
    pr_ratio =  market_cap/revenues
    if pr_ratio>=200:
        return None 
    return pr_ratio

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
  
def get_historical_market_cap(date,stock,sorted_data=None,do_api_call=True):
    if date != 'all':
        if do_api_call ==False and sorted_data is None:
            return None
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


def get_stock_price_at_date(symbol, date,historical_data =None,not_None=False):
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()

    if historical_data is not None:
        if not_None==True:
           #if Nnot_None is True i want to get the most recent price before date
           for days_back in range(4): 
                test_date = date - timedelta(days=days_back)
                if test_date in historical_data:
                    return historical_data[test_date]
           return None  # If no data found in the last 4 days
        return historical_data.get(date)
    else:
       return None
    
def get_SMA(symbol, date, period,length,sorted_data=None):
    if date != 'all':
        if isinstance(date, str):
            date = datetime.strptime(date, '%Y-%m-%d').date()
        elif isinstance(date, pd.Timestamp):
            date = date.date()
        elif isinstance(date, datetime):
            date = date.date()

    if sorted_data==None:
        url_sma = f'https://financialmodelingprep.com/api/v3/technical_indicator/{period}/{symbol}?type=sma&period={length}&apikey={api_key}'
        try:
            response_sma = requests.get(url_sma, timeout=1)
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
    
    for entry in sorted_data:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if entry_date <= date:
            return entry['sma']
    
    return None

def extract_other_expenses(date,sorted_data_income):
    if sorted_data_income is None:
        return None

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d")
        if entry_date <= date:
            return entry['otherExpenses']
    return None


def extract_other_expenses(date,sorted_data_income):
    if sorted_data_income is None:
        return None

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d")
        if entry_date <= date:
            return entry['otherExpenses']
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


def extract_eps(date,sorted_data_income):
    if sorted_data_income is None:
        return None

    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    elif isinstance(date, datetime):
        date = date.date()

    min_date = date - timedelta(days=370)

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date() 
        if min_date <= entry_date <= date and entry['eps'] is not None and entry['eps'] !=0:
            return entry['eps']
        
    return None

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

def extract_interestCoverage(date,sorted_data_income):
    if sorted_data_income is None:
        return None

    min_date = date - timedelta(days=370)

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d")
        if min_date <= entry_date <= date and entry['operatingIncome'] is not None and entry['interestExpense'] is not None and entry['interestExpense'] !=0:
            return entry['operatingIncome']/entry['interestExpense']
        
    return None