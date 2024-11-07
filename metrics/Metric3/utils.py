import pandas as pd
from datetime import datetime, timedelta ,date
import random
from typing import List, Dict,Union
import requests
import matplotlib.pyplot as plt
from typing import Optional
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

def parse_date(date_value: Union[str, date]) -> datetime:
    """
    Helper function to parse different date formats into datetime object.
    
    Parameters:
    date_value (Union[str, date]): Date in string format 'YYYY-MM-DD' or date object
    
    Returns:
    datetime: Parsed datetime object
    """
    if isinstance(date_value, date):
        return datetime.combine(date_value, datetime.min.time())
    return datetime.strptime(date_value, '%Y-%m-%d')

def fetch_stock_data(stocks):
    print("fetch_stock_data...")
    historical_data_for_stock = {}
    market_cap_dict = {}
    revenues_dict = {}
    hist_data_df_for_stock = {}
    balance_shit_dict= {}
    for c, stock in enumerate(stocks):
        historic_data = get_historical_price(stock)
        date_to_close = convert_to_dict(historic_data)
        hist_data_df_for_stock[stock] = convert_to_df(date_to_close)
        historical_data_for_stock[stock] = date_to_close

        market_cap_dict[stock] = get_historical_market_cap('all',stock,sorted_data=None)
        revenues_dict[stock] = get_income_dict(stock,'quarter') #should be named get_income_statementss
        balance_shit_dict[stock] = get_balance_shit(stock,'quarter')
        if c % int(-0.2*len(stocks)+140) == 0 and c > 0:
            print("we sleep")
            time.sleep(50)

    return historical_data_for_stock, market_cap_dict, revenues_dict, hist_data_df_for_stock,balance_shit_dict

def get_cashflow_dict(symbol,period):
    url_income = f'https://financialmodelingprep.com/api/v3/cash-flow-statement/{symbol}?period={period}&apikey={api_key}'

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

def get_income_dict(symbol,period):
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

def get_balance_shit(symbol,period):
    balance_sheet_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{symbol}?period={period}&apikey={api_key}"
    try:
        response_income = requests.get(balance_sheet_url, timeout=2)
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None

    if response_income.status_code != 200:
        print(f"Error: API returned status code {response_income.status_code}")
        return None
    data_income = response_income.json()
    sorted_data = sorted(data_income, key=lambda x: datetime.strptime(x['date'].split()[0], "%Y-%m-%d"), reverse=True)

    return sorted_data

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

def convert_to_df(date_to_close):
    if date_to_close is None:
       return None
    date_to_close_df = pd.DataFrame(date_to_close.items(), columns=['date', 'close'])
    date_to_close_df['date'] = pd.to_datetime(date_to_close_df['date'])
    date_to_close_df.set_index('date', inplace=True)
    date_to_close_df.sort_index(inplace=True)
    return date_to_close_df

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
    

def get_data_for_date(data: list, target_date: str) -> Optional[dict]:
    """
    Helper function to get the most recent data point before or equal to the target date.
    
    Parameters:
    data (list): List of financial data dictionaries
    target_date (str): Target date in 'YYYY-MM-DD' format
    
    Returns:
    dict: Most recent data point before or equal to target date, or None if not found
    """
    target_datetime = parse_date(target_date)
    
    # Sort data by date in descending order
    """
    sorted_data = sorted(data, 
                        key=lambda x: parse_date(x['date']), 
                        reverse=True)
    """
    sorted_data = data

    min_date = target_datetime - timedelta(days=100)

    # Find the most recent data point before or equal to target date
    for item in sorted_data:
        item_date = parse_date(item['date'])
        if min_date<= item_date <= target_datetime:
            return item
            
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
                return round(entry['sma'],2)
    
    return None

def fetch_data_for_ml(stocks,income_mc=False):
    sma_10d_dict = {}
    sma_50w_dict = {}
    sma_100d_dict = {}
    sma_200d_dict = {}
    sma_50d_dict = {}
    income_dict = {}
    market_cap_dict = {}
    cashflow_dict = {}
    for i,stock in enumerate(stocks):
        sma_10d_dict[stock]=get_SMA(stock, 'all', '1day', 10,use_api_call=True)
        sma_50w_dict[stock]=get_SMA(stock, 'all', '1week', 50,use_api_call=True)
        sma_100d_dict[stock]=get_SMA(stock, 'all', '1day', 100,use_api_call=True)
        sma_200d_dict[stock]=get_SMA(stock, 'all', '1day', 200,use_api_call=True)
        sma_50d_dict[stock]=get_SMA(stock, 'all', '1day', 50,use_api_call=True)
        if income_mc:
            income_dict[stock]=get_income_dict(stock, 'quarter')
            market_cap_dict[stock] = get_historical_market_cap('all',stock,sorted_data=None)
        cashflow_dict[stock]= get_cashflow_dict(stock,'quarter')
        if i%20==0:
            print(f"i = {i} we sleep")
            time.sleep(20)
    return sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,income_dict,market_cap_dict,cashflow_dict


def load_models():
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric3/models/rf_100_4_6.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric3/models/scaler_4_6.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric3/models/xg_c4_14_8.pkl', 'rb') as model_file:
        model2 = pickle.load(model_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric3/models/scaler_14_8.pkl', 'rb') as scaler_file:
        scaler2 = pickle.load(scaler_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric3/models/rf_200_12_6.pkl', 'rb') as model_file:
        model3 = pickle.load(model_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric3/models/scaler_12_6.pkl', 'rb') as scaler_file:
        scaler3 = pickle.load(scaler_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric3/models/xgb200_model_21_10.pkl', 'rb') as model_file:
        model4 = pickle.load(model_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric3/models/scaler_21_10.pkl', 'rb') as scaler_file:
        scaler4 = pickle.load(scaler_file)
    return model,scaler,model2,scaler2,model3,scaler3,model4,scaler4


def safe_divide(a, b):
    return round(a / b, 2) if b != 0 else None

def predict_buy(model,scaler,date,symbol,ratio,features_for_pred,price,sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict ,market_cap_dict,cashflow_dict):
    if income_dict is not None and not isinstance(income_dict, dict):
        eps = extract_eps_from_income_statement(date, income_dict)
        other_expenses = extract_other_expenses(date, income_dict)
        fcf = extract_fcf(date, cashflow_dict)
        #ebitdaMargin = extract_ebitda_margin(date, income_dict)
        #revenues,_= extract_revenue(date, income_dict)
        ocfToNetinc = extract_ocf_to_net_income(date, cashflow_dict)

    elif isinstance(income_dict, dict):
        eps = extract_eps_from_income_statement(date, income_dict[symbol])
        other_expenses=extract_other_expenses(date, income_dict[symbol])
        fcf = extract_fcf(date, cashflow_dict[symbol])
        ocfToNetinc = extract_ocf_to_net_income(date, cashflow_dict[symbol])

        #ebitdaMargin = extract_ebitda_margin(date, income_dict[symbol])
        #revenues,_= extract_revenue(date, income_dict[symbol])

    feature_calculations = {
        'MarkRevRatio': lambda: calculate_mark_rev_ratio(date, symbol, market_cap_dict, income_dict),
        'sma_50w': lambda: get_SMA(symbol, date, '1week', 50, sorted_data=None if sma_50w_dict is None else sma_50w_dict[symbol],use_api_call=True),
        'sma_200d': lambda: get_SMA(symbol, date, '1day', 200, sorted_data=None if sma_200d_dict is None else sma_200d_dict[symbol],use_api_call=True),
        'sma_100d': lambda: get_SMA(symbol, date, '1day', 100, sorted_data=None if sma_100d_dict is None else sma_100d_dict[symbol],use_api_call=True),
        'sma_10d': lambda: get_SMA(symbol, date, '1day', 10, sorted_data=None if sma_10d_dict is None else sma_10d_dict[symbol],use_api_call=True),
        'price': lambda: price,
        'ratio': lambda: ratio,
        'other_expenses': lambda:other_expenses,
        'fcf': lambda:fcf,
        'peRatio': lambda: price / eps if price is not None and eps is not None and eps != 0 else None,
        'ocfToNetinc': lambda:ocfToNetinc,
        'price_to_50w': lambda: price / feature_calculations['sma_50w']() if price is not None and feature_calculations['sma_50w']() is not None and feature_calculations['sma_50w']() != 0 else None
    }

    # Calculate only the required features
    data_dict = {'date': date, 'symbol': symbol}
    for feature in features_for_pred:
        if feature in feature_calculations:
            data_dict[feature] = feature_calculations[feature]()

    data = pd.DataFrame([data_dict])


    if 'peRatio' in data:
        data['peRatio'] = data['peRatio'].round(2)
    #print(data)
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
            return round(entry['eps'], 2)
        
    return None


def extract_cashAtEndOfPeriod(date,sorted_data):
    if sorted_data is None:
        return None

    for entry in sorted_data:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if entry_date <= date:
            return entry['cashAtEndOfPeriod']
    return None


def extract_other_expenses(input_date,sorted_data_income):
    if sorted_data_income is None:
        return None

    if isinstance(input_date, date):
        input_date = datetime.combine(input_date, datetime.min.time())
    if isinstance(input_date, str):
        input_date = datetime.strptime(input_date, '%Y-%m-%d').date()
    elif isinstance(input_date, pd.Timestamp):
        input_date = input_date.date()
    elif isinstance(input_date, datetime):
        input_date = input_date.date()

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if entry_date <= input_date:
            return round(entry['otherExpenses'], 2)
    return None


def extract_fcf(input_date,sorted_data):
    if sorted_data is None:
        return None

    if isinstance(input_date, date):
        input_date = datetime.combine(input_date, datetime.min.time())
    if isinstance(input_date, str):
        input_date = datetime.strptime(input_date, '%Y-%m-%d').date()
    elif isinstance(input_date, pd.Timestamp):
        input_date = input_date.date()
    elif isinstance(input_date, datetime):
        input_date = input_date.date()

    min_date = input_date - timedelta(days=100)

    for entry in sorted_data:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if min_date <= entry_date <= input_date :
            return round(entry['operatingCashFlow'] - entry['capitalExpenditure'],2)
        
    return None

def extract_ocf_to_net_income(input_date,sorted_data):
    if sorted_data is None:
        return None

    if isinstance(input_date, date):
        input_date = datetime.combine(input_date, datetime.min.time())
    if isinstance(input_date, str):
        input_date = datetime.strptime(input_date, '%Y-%m-%d').date()
    elif isinstance(input_date, pd.Timestamp):
        input_date = input_date.date()
    elif isinstance(input_date, datetime):
        input_date = input_date.date()

    min_date = input_date - timedelta(days=120)

    for entry in sorted_data:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if min_date <= entry_date <= input_date and entry['netIncome'] is not None and entry['operatingCashFlow'] is not None and entry['netIncome'] !=0:
            return entry['operatingCashFlow']/entry['netIncome']
        
    return None

def extract_net_profit_margin(input_date,sorted_data_income):
    if sorted_data_income is None:
        return None

    if isinstance(input_date, date):
        input_date = datetime.combine(input_date, datetime.min.time())
    if isinstance(input_date, str):
        input_date = datetime.strptime(input_date, '%Y-%m-%d').date()
    elif isinstance(input_date, pd.Timestamp):
        input_date = input_date.date()
    elif isinstance(input_date, datetime):
        input_date = input_date.date()

    min_date = input_date - timedelta(days=120)

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if min_date <= entry_date <= input_date and entry['revenue'] is not None and entry['netIncome'] is not None and entry['revenue'] !=0:
            return entry['netIncome']/entry['revenue']
        
    return None


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
        return round((market_cap / revenue),2)
    else:
        return None
    
def extract_costOfRevenue(date,sorted_data_income):
    if sorted_data_income is None:
        return None

    min_date = date - timedelta(days=370)

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if min_date <= entry_date <= date and entry['costOfRevenue'] is not None:
            return entry['costOfRevenue']
        
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

def calculate_evgp_ratio(ticker: str,market_cap: float, target_date: str,balance_sheet_data,sorted_income_stmt_data,do_api_call=True) -> Optional[float]:
    """
    Calculate the Enterprise Value to Gross Profit ratio for a given stock ticker.
    
    Parameters:
    ticker (str): Stock ticker symbol (e.g., 'AAPL')
    api_key (str): API key for Financial Modeling Prep API
    market_cap (float): Current market capitalization of the company
    
    Returns:
    float: EV/GP ratio, or None if data cannot be retrieved
    """
    try:
        if balance_sheet_data is None and do_api_call:
            # Get balance sheet data
            balance_sheet_url = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{ticker}?period=quarter&apikey={api_key}"
            balance_sheet_response = requests.get(balance_sheet_url)
            balance_sheet_data = balance_sheet_response.json()
        
        if sorted_income_stmt_data is None and do_api_call:
            # Get income statement data
            income_stmt_url = f"https://financialmodelingprep.com/api/v3/income-statement/{ticker}?period=quarter&apikey={api_key}"
            income_stmt_response = requests.get(income_stmt_url)
            income_stmt_data = income_stmt_response.json()
            sorted_income_stmt_data = sorted(income_stmt_data, key=lambda x: datetime.strptime(x['date'].split()[0], "%Y-%m-%d"), reverse=True)

            
        if not balance_sheet_data or not sorted_income_stmt_data:
            return None
            
        latest_balance_sheet = get_data_for_date(balance_sheet_data, target_date)
        latest_income_stmt = get_data_for_date(sorted_income_stmt_data, target_date)
        

        if latest_balance_sheet is None or latest_income_stmt is None:
            return None
        # Calculate Enterprise Value
        total_debt = latest_balance_sheet['totalDebt']
        cash_and_equivalents = latest_balance_sheet['cashAndCashEquivalents']
        if market_cap is None or total_debt is None or cash_and_equivalents is None:
            return None
        enterprise_value = market_cap + total_debt - cash_and_equivalents
        
        # Calculate Gross Profit (already provided in income statement)
        gross_profit = latest_income_stmt['grossProfit']
        
        if gross_profit is None or gross_profit==0:
            return None
        # Calculate EV/GP ratio
        ev_gp_ratio = enterprise_value / gross_profit
        if ev_gp_ratio>100 or ev_gp_ratio<0:
            return None
        return ev_gp_ratio
        
    except Exception as e:
        print(f"Error calculating EV/GP ratio: {str(e)}")
        return None
    

def get_max_price_in_range(symbol, start_date, end_date,historical_data=None,do_api_call=True):
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

    if do_api_call:
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