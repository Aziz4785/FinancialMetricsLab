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

def fetch_stock_data(stocks):
    print("fetch_stock_data...")
    historical_data_for_stock = {}
    market_cap_dict = {}
    revenues_dict = {}
    hist_data_df_for_stock = {}
    balance_shit_dict= {}
    cashflow_dict = {}
    estimations_dict = {}
    for c, stock in enumerate(stocks):
        print(c)
        historic_data = get_historical_price(stock)
        date_to_close = convert_to_dict(historic_data)
        hist_data_df_for_stock[stock] = convert_to_df(date_to_close)
        historical_data_for_stock[stock] = date_to_close

        #market_cap_dict[stock] = get_historical_market_cap(stock)
        revenues_dict[stock] = get_income_dict(stock,'quarter')
        #balance_shit_dict[stock] = get_balance_shit(stock,'quarter')
        cashflow_dict[stock]= get_cashflow_dict(stock,'quarter')
        estimations_dict[stock] = get_estimation_dict(stock,'quarter')
        if estimations_dict[stock] is None:
            print("estimations dict is none for : ",stock)
        if c % 40==0 and c > 0:
            print("we sleep")
            time.sleep(50)

    return historical_data_for_stock, market_cap_dict, revenues_dict, hist_data_df_for_stock,balance_shit_dict,cashflow_dict,estimations_dict

def extract_net_inome(input_date,sorted_data_income):
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

    min_date = input_date - timedelta(days=100)

    for entry in sorted_data_income:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if min_date <= entry_date <= input_date:
            return entry['netIncome']
        
    return None

def extract_dividend_paid(input_date,sorted_cashflow_data):
    if sorted_cashflow_data is None:
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

    for entry in sorted_cashflow_data:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if min_date <= entry_date <= input_date:
            return entry['dividendsPaid']
        
    return None

def calculate_revenue_growth(input_date,estimations_data):
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

    if estimations_data is None:
        return None
    input_date = ensure_datetime(input_date)
    
    current_estimated_eps = None
    future_estimated_eps = None
    current_estim_rev = None
    future_estim_rev = None

    for i, entry in enumerate(estimations_data):
        if entry is not None:
            entry_date = ensure_datetime(entry['date'])

            if entry_date <= input_date:
                current_estim_rev = entry['estimatedRevenueAvg']       
                if i > 0:
                    future_estim_rev = estimations_data[i-1]['estimatedRevenueAvg']
                break
    if current_estim_rev is not None and future_estim_rev is not None and current_estim_rev!=0:
        return (future_estim_rev-current_estim_rev)/current_estim_rev
    return None

def calculate_div_payout_ratio(input_date,cashflow_data,income_data):
    div_paid = extract_dividend_paid(input_date,cashflow_data)
    net_income = extract_net_inome(input_date,income_data)

    if div_paid is None or net_income is None or net_income ==0:
        return None 
    return div_paid/net_income

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


def extract_income_features(input_date,sorted_income_data):
    if isinstance(input_date, str):
        input_date = datetime.strptime(input_date, '%Y-%m-%d').date()
    elif isinstance(input_date, pd.Timestamp):
        input_date = input_date.date()
    elif isinstance(input_date, datetime):
        input_date = input_date.date()

    if sorted_income_data==None:
        return None
    
    min_date = input_date - timedelta(days=100)

    for entry in sorted_income_data:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if min_date<=entry_date <= input_date:
            lag_days = (input_date - entry_date).days
            return {
                "revenue": entry["revenue"],
                "inc_lag_days": lag_days,
                "costOfRevenue": entry["costOfRevenue"],
                "grossProfit": entry["grossProfit"],
                "netIncome": entry["netIncome"],
                "otherExpenses": entry["otherExpenses"],
                "eps": entry["eps"],
                "ebitda": entry["ebitda"],
                "researchAndDevelopmentExpenses": entry["researchAndDevelopmentExpenses"],
                "interestIncome": entry["interestIncome"]
            }
    return None


def extract_estimation_features(input_date,sorted_estimation_data):
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

    if sorted_estimation_data is None:
        return None
    input_date = ensure_datetime(input_date)
    
    current_estimated_eps = None
    future_estimated_eps = None
    current_estim_rev = None
    future_estim_rev = None

    for i, entry in enumerate(sorted_estimation_data):
        if entry is not None:
            entry_date = ensure_datetime(entry['date'])

            if entry_date <= input_date:
                if current_estimated_eps is None:
                    current_estimated_eps = entry['estimatedEpsAvg']
                    current_estim_rev = entry['estimatedRevenueAvg']
                
                if i > 0:
                    future_estimated_eps = sorted_estimation_data[i-1]['estimatedEpsAvg']
                    future_estim_rev = sorted_estimation_data[i-1]['estimatedRevenueAvg']
                break

    if current_estimated_eps is not None and future_estimated_eps is not None:    
        return {
                    "current_estimated_eps": current_estimated_eps,
                    "future_estimated_eps": future_estimated_eps,
                    "current_estim_rev": current_estim_rev,
                    "future_estim_rev": future_estim_rev
                }
    else :
        return None
    
def extract_cashflow_features(input_date,sorted_cashflow_data):
    if isinstance(input_date, str):
        input_date = datetime.strptime(input_date, '%Y-%m-%d').date()
    elif isinstance(input_date, pd.Timestamp):
        input_date = input_date.date()
    elif isinstance(input_date, datetime):
        input_date = input_date.date()

    if sorted_cashflow_data==None:
        return None
    
    min_date = input_date - timedelta(days=100)

    for entry in sorted_cashflow_data:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if min_date<=entry_date <= input_date:
            lag_days = (input_date - entry_date).days
            return {
                "debtRepayment": entry["debtRepayment"],
                "lag_days": lag_days,
                "freeCashFlow": entry["freeCashFlow"],
                "dividendsPaid": entry["dividendsPaid"],
                "otherFinancingActivites": entry["otherFinancingActivites"],
                "operatingCashFlow": entry['operatingCashFlow'],
                "capitalExpenditure": entry['capitalExpenditure']
            }
    return None

def extract_balance_features(input_date,sorted_BALANCE_data):
    if isinstance(input_date, str):
        input_date = datetime.strptime(input_date, '%Y-%m-%d').date()
    elif isinstance(input_date, pd.Timestamp):
        input_date = input_date.date()
    elif isinstance(input_date, datetime):
        input_date = input_date.date()

    if sorted_BALANCE_data==None:
        return None
    
    min_date = input_date - timedelta(days=100)

    for entry in sorted_BALANCE_data:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if min_date<=entry_date <= input_date:
            lag_days = (input_date - entry_date).days
            return {
                "cashAndCashEquivalents": entry["cashAndCashEquivalents"],
                "totalDebt": entry["totalDebt"],
                "bal_lag_days":lag_days,
                "netDebt": entry["netDebt"],
                "totalAssets": entry["totalAssets"],
                "otherCurrentAssets": entry["otherCurrentAssets"],
                "totalLiabilities": entry["totalLiabilities"],
                "totalEquity": entry["totalEquity"],
                "cashAndShortTermInvestments": entry["cashAndShortTermInvestments"],
                "netReceivables": entry["netReceivables"],
                "totalCurrentLiabilities": entry["totalCurrentLiabilities"],
                "totalStockholdersEquity": entry["totalStockholdersEquity"]
            }
    return None


def extract_market_cap(date,sorted_data):

    if sorted_data is None:
        return None
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    elif isinstance(date, datetime):
        date = date.date()

    if sorted_data==None:
        return None
    
    min_date = date - timedelta(days=5)

    for entry in sorted_data:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if min_date<=entry_date <= date:
            return entry['marketCap']
    
    return None


def extract_max_price_in_range(start_date, end_date,historical_data):
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

    return None

def extract_stock_price_at_date(date,historical_data =None,not_None=False):
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

def get_estimation_dict(symbol,period):
    url = f'https://financialmodelingprep.com/api/v3/analyst-estimates/{symbol}?period={period}&apikey={api_key}'
    try:
        response_income = requests.get(url, timeout=2)
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None

    if response_income.status_code != 200:
        print(f"Error: API returned status code {response_income.status_code}")
        return None
    data_income = response_income.json()
    sorted_data = sorted(data_income, key=lambda x: datetime.strptime(x['date'].split()[0], "%Y-%m-%d"), reverse=True)

    return sorted_data

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

def get_historical_market_cap(stock):
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

    
    return sorted_data


def convert_to_df(date_to_close):
    if date_to_close is None:
       return None
    date_to_close_df = pd.DataFrame(date_to_close.items(), columns=['date', 'close'])
    date_to_close_df['date'] = pd.to_datetime(date_to_close_df['date'])
    date_to_close_df.set_index('date', inplace=True)
    date_to_close_df.sort_index(inplace=True)
    return date_to_close_df

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