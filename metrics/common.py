import pandas as pd
from datetime import datetime, timedelta ,date
from requests.exceptions import Timeout ,ConnectionError,RequestException
import requests
import os
import random
from dotenv import load_dotenv
load_dotenv()

# Access the API key
api_key = os.getenv('FMP_API_KEY')

def load_stocks(nbr,path='sp500_final.csv'):
    stocks_df = pd.read_csv(path)
    stocks = stocks_df['Ticker'].tolist()
    stocks = random.sample(stocks, nbr)
    return stocks


def unique_stocks_threshold(nbr_stocks,nbr_dates):
    if abs(nbr_stocks-250)<=3 and abs(nbr_dates-500)<=3:
        return 14,31
    elif abs(nbr_stocks-200)<=3 and nbr_dates==500:
        return 12,29
    elif nbr_stocks==500 and nbr_dates==1000:
        return 40,60
    elif nbr_stocks==100 and nbr_dates==100:
        return 8,11
    return 14,31


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


def extract_Field(date,sorted_source_data,field_name):
    if sorted_source_data is None:
        return None

    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    elif isinstance(date, datetime):
        date = date.date()

    min_date = date - timedelta(days=100)

    for entry in sorted_source_data:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date() 
        if min_date <= entry_date <= date:
            if field_name in entry:
                return round(entry[field_name], 2)
    print("we found nothing !")
    return None

def safe_divide(numerator, denominator):
    try:
        if denominator == 0 or numerator is None or denominator is None:
            return None
        return numerator / denominator
    except (ZeroDivisionError, TypeError):
        return None
    
def get_current_price(symbol):
    url = f'https://financialmodelingprep.com/api/v3/quote/{symbol}?apikey={api_key}'
    try:
        response = requests.get(url, timeout=2)
        data_price = response.json()
        return data_price[0].get('price') if data_price else None
    except (Timeout, requests.RequestException, IndexError, KeyError):
        return None


def convert_to_df(date_to_close):
    if date_to_close is None:
       return None
    date_to_close_df = pd.DataFrame(date_to_close.items(), columns=['date', 'close'])
    date_to_close_df['date'] = pd.to_datetime(date_to_close_df['date'])
    date_to_close_df.set_index('date', inplace=True)
    date_to_close_df.sort_index(inplace=True)
    return date_to_close_df

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

def safe_dict_get(dictionary, key, default=None):
        if dictionary is None:
            return default
        return dictionary.get(key, default)
   
def safe_subtract(a, b):
    if a is None or b is None:
        return None
    return a - b

def safe_add(a, b):
    if a is None or b is None:
        return None
    return a + b


def safe_multiply(a, b):
    if a is None or b is None:
        return None
    return a * b


def calculate_EV(mc,balance_features,income_features):
    if balance_features is None or income_features is None:
        return None
    total_debt=balance_features['totalDebt']
    cashncasheq = balance_features['cashAndCashEquivalents']
    if mc is None or total_debt is None :
        return None
    
    return mc+total_debt-cashncasheq

def get_historical_price(symbol):
    url_price = f'https://financialmodelingprep.com/api/v3/historical-chart/1day/{symbol}?from=2019-08-08&to=2024-11-27&apikey={api_key}'
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
    
    min_date = date - timedelta(days=3)

    for entry in sorted_data:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
        if min_date<=entry_date <= date:
            return entry['marketCap']
    
    return None

def get_company_sector(stock):
    url = f'https://financialmodelingprep.com/api/v3/profile/{stock}?apikey={api_key}'

    try:
        response_= requests.get(url, timeout=2)
    except requests.exceptions.Timeout:
        print("Request timed out")
        return None

    if response_.status_code != 200:
        print(f"Error: API returned status code {response_.status_code}")
        return None
    
    data = response_.json()
    if not data or not isinstance(data, list) or 'sector' not in data[0]:
        print("Error: Invalid or empty response data")
        return "unknown"  # Default value if sector is missing

    return data[0]['sector'].lower() if data[0]['sector'] else "unknown"


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

def calculate_pe_from_features(price_at_date,income_features):
    if income_features is None or price_at_date is None:
        return None 
    
    if price_at_date <=0:
        return None 
    
    return safe_divide(price_at_date,income_features['eps'])

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


def extract_SMA(date,sorted_data):
    if sorted_data is None:
        return None 
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    elif isinstance(date, datetime):
        date = date.date()

    min_date = date - timedelta(days=10)

    if sorted_data is not None:
        for entry in sorted_data:
            entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
            if min_date <= entry_date <= date:
                return round(entry['sma'],2)
    
    return None