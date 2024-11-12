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
    sector_dict = {}
    for c, stock in enumerate(stocks):
        print(c)
        historic_data = get_historical_price(stock)
        date_to_close = convert_to_dict(historic_data)
        hist_data_df_for_stock[stock] = convert_to_df(date_to_close)
        historical_data_for_stock[stock] = date_to_close
        sector_dict[stock]=get_company_sector(stock)
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

    return historical_data_for_stock, market_cap_dict, revenues_dict, hist_data_df_for_stock,balance_shit_dict,cashflow_dict,estimations_dict,sector_dict


def engineer_stock_features(df):
    """
    Create engineered features for stock prediction using cross-sectional data
    
    Parameters:
    df (pd.DataFrame): DataFrame containing different stocks' features at a single point in time
    
    Returns:
    pd.DataFrame: DataFrame with additional engineered features
    """
    # Create a copy to avoid modifying the original DataFrame
    df_new = df.copy()
    
    # 1. Relative Price Positions
    sma_columns_all = ['sma_10d', 'sma_50d', 'sma_100d', 'sma_200d']
    sma_columns=[]
    for sma in sma_columns_all:
        if sma in df_new.columns:
            sma_columns.append(sma)
    # Price relative to SMAs
    for sma in sma_columns:
        if sma in df_new.columns:
            df_new[f'price_to_{sma}_ratio'] = df_new['price'] / df_new[sma] - 1
    
    # Gaps between SMAs
    for i in range(len(sma_columns)):
        for j in range(i+1, len(sma_columns)):
            sma1, sma2 = sma_columns[i], sma_columns[j]
            if sma1 in df_new.columns and sma2 in df_new.columns:
                df_new[f'{sma1}_to_{sma2}_ratio'] = df_new[sma1] / df_new[sma2] - 1
    
    # 2. Technical Position Indicators
    # SMA Cross signals
    if 'sma_200d' in df_new.columns and 'sma_50d' in  df_new.columns:
        df_new['golden_cross'] = (df_new['sma_50d'] > df_new['sma_200d']).astype(int)
        df_new['death_cross'] = (df_new['sma_50d'] < df_new['sma_200d']).astype(int)
    
    # Count how many SMAs the price is above
    sma_comparisons = [df_new['price'] > df_new[sma] for sma in sma_columns]
    df_new['smas_above_count'] = sum(sma_comparisons)
    
    # Boolean indicators for price position
    df_new['price_above_all_sma'] = (df_new['smas_above_count'] == len(sma_columns)).astype(int)
    df_new['price_below_all_sma'] = (df_new['smas_above_count'] == 0).astype(int)
    
    # 3. Valuation Ratios Combinations
    valuation_metrics = ['pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'markRevRAtio']
    
    # Normalize valuation metrics (handling potential missing values)
    for metric in valuation_metrics:
        if metric in df_new.columns:
            df_new[f'{metric}_normalized'] = (
                df_new[metric]
                .pipe(lambda x: (x - x.mean()) / x.std())
            )
    
    # Combined valuation score (equal weights)
    normalized_columns = [col for col in df_new.columns if col.endswith('_normalized')]
    if normalized_columns:
        df_new['combined_valuation_score'] = df_new[normalized_columns].mean(axis=1)
    
    # 4. Size and Value Combinations
    # Market cap quintile (1 = smallest, 5 = largest)
    df_new['market_cap_quintile'] = pd.qcut(df_new['marketCap'], q=5, labels=[1,2,3,4,5], duplicates='drop')
    
    # Value score combining different metrics (handling missing values)
    value_metrics = []
    if 'pe_normalized' in df_new.columns:
        value_metrics.append('pe_normalized')
    if 'PriceToSales_normalized' in df_new.columns:
        value_metrics.append('PriceToSales_normalized')
    
    if value_metrics:
        df_new['value_score'] = df_new[value_metrics].mean(axis=1)
    
    # 5. Historical Performance vs Current Valuation
    # Compare 3M return with valuation metrics (handling division by zero and infinities)
    if 'pe' in df_new.columns:
        df_new['return_to_pe'] = df_new['3M_return'] / df_new['pe'].replace(0, np.nan)
    if 'markRevRAtio' in df_new.columns:
        df_new['return_to_ps'] = df_new['3M_return'] / df_new['markRevRAtio'].replace(0, np.nan)
    
    # 6. Year-over-Year Growth
    if 'sma_10d_11months_ago' in df_new.columns:
        df_new['sma10_yoy_growth'] = (df_new['sma_10d'] / df_new['sma_10d_11months_ago'] - 1) * 100
    
    # 7. Complex Ratios
    # Enterprise value ratios combinations
    ev_metrics = []
    if 'EVEbitdaRatio_normalized' in df_new.columns:
        ev_metrics.append('EVEbitdaRatio_normalized')
    if 'EVGP_normalized' in df_new.columns:
        ev_metrics.append('EVGP_normalized')
    
    if ev_metrics:
        df_new['ev_composite'] = df_new[ev_metrics].mean(axis=1)

    
    return df_new

def fetch_data_for_ml(stocks):
    sma_10d_dict = {}
    sma_50w_dict = {}
    sma_100d_dict = {}
    sma_200d_dict = {}
    sma_50d_dict = {}
    market_cap_dict = {}
    balance_dict = {}
    std_10d_dict={}
    for i,stock in enumerate(stocks):
        sma_10d_dict[stock]=get_SMA(stock, '1day', 10)
        #sma_50w_dict[stock]=get_SMA(stock,  '1week', 50)
        sma_100d_dict[stock]=get_SMA(stock, '1day', 100)
        sma_200d_dict[stock]=get_SMA(stock, '1day', 200)
        sma_50d_dict[stock]=get_SMA(stock, '1day', 50)
        market_cap_dict[stock] = get_historical_market_cap(stock)
        balance_dict[stock] = get_balance_shit(stock,'quarter')
        std_10d_dict[stock] = get_std(stock,'1day',10)
        if i%20==0:
            print(f"i = {i} we sleep")
            time.sleep(20)
    return sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,market_cap_dict,balance_dict,std_10d_dict

def get_SMA(symbol,period,length):

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

    return sorted_data


def get_std(symbol,period,length):

    url_sma = f'https://financialmodelingprep.com/api/v3/technical_indicator/{period}/{symbol}?type=standardDeviation&period={length}&apikey={api_key}'
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

    return sorted_data


def add_historical_sma(df: pd.DataFrame, sma_10d_dict: Dict) -> pd.DataFrame:

    periods = [
        ('weeks', 1),
        ('months', 1),
        ('months', 2),
        ('months', 3),
        ('months', 4),
        ('months', 5),
        ('months', 6),
        ('months', 7),
        ('months', 11)
    ]
    
    # Create a date matrix for all calculations at once
    dates_matrix = {}
    for unit, value in periods:
        col_name = f'sma_10d_{value}{unit[0]}_ago'  # w for weeks, m for months
        dates_matrix[col_name] = pd.to_datetime(df['date']) - pd.DateOffset(**{unit: value})
    
    # Vectorized calculation of all SMA columns
    for col_name, dates in dates_matrix.items():
        # Create a list of (date, symbol) tuples for efficient lookup
        lookup_data = list(zip(dates.dt.date, df['symbol']))
        
        # Use list comprehension for faster processing
        df[col_name] = [
            extract_SMA(date, sma_10d_dict.get(symbol))
            for date, symbol in lookup_data
        ]
    
    return df

def extract_SMA(date,sorted_data):
    if sorted_data is None:
        return None 
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    elif isinstance(date, datetime):
        date = date.date()

    if sorted_data is not None:
        for entry in sorted_data:
            entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
            if entry_date <= date:
                return round(entry['sma'],2)
    
    return None


def extract_STD(date,sorted_data):
    if sorted_data is None:
        return None 
    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    elif isinstance(date, datetime):
        date = date.date()

    if sorted_data is not None:
        for entry in sorted_data:
            entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date()
            if entry_date <= date:
                return round(entry['standardDeviation'],2)
    
    return None

def extract_multiple_SMAs(base_date: Union[str, datetime, pd.Timestamp], sorted_data: List[dict]) -> Dict[str, float]:
    if sorted_data is None:
        return {f'sma_10d_{i}months_ago': None for i in [1,2,3,4,5,6,7,11]}
    
    # Convert base_date to date object
    if isinstance(base_date, str):
        base_date = datetime.strptime(base_date, '%Y-%m-%d').date()
    elif isinstance(base_date, pd.Timestamp):
        base_date = base_date.date()
    elif isinstance(base_date, datetime):
        base_date = base_date.date()
    
    # Calculate all target dates upfront
    target_dates = {
        'sma_10d_1weeks_ago': base_date - pd.DateOffset(weeks=1),
        'sma_10d_1months_ago': base_date - pd.DateOffset(months=1),
        'sma_10d_2months_ago': base_date - pd.DateOffset(months=2),
        'sma_10d_3months_ago': base_date - pd.DateOffset(months=3),
        'sma_10d_4months_ago': base_date - pd.DateOffset(months=4),
        'sma_10d_5months_ago': base_date - pd.DateOffset(months=5),
        'sma_10d_6months_ago': base_date - pd.DateOffset(months=6),
        'sma_10d_7months_ago': base_date - pd.DateOffset(months=7),
        'sma_10d_11months_ago': base_date - pd.DateOffset(months=11)
    }
    labels = ['sma_10d_1weeks_ago','sma_10d_1months_ago','sma_10d_2months_ago','sma_10d_3months_ago','sma_10d_4months_ago','sma_10d_5months_ago','sma_10d_6months_ago','sma_10d_7months_ago', 'sma_10d_11months_ago']
    date_list = [base_date - pd.DateOffset(weeks=1),base_date - pd.DateOffset(months=1),base_date - pd.DateOffset(months=2),base_date - pd.DateOffset(months=3), base_date - pd.DateOffset(months=4),base_date - pd.DateOffset(months=5), base_date - pd.DateOffset(months=6), base_date - pd.DateOffset(months=7), base_date - pd.DateOffset(months=11)]
    date_list = [d.date() for d in date_list] 
    #target_dates = {k: v.date() for k, v in target_dates.items()}
    #print("labels : ")
    #print(labels)
    #print("date_list:")
    #print(date_list)
    results = {k: None for k in labels}
    
    current_target_idx = 0
    
    # Single pass through the sorted data
    for entry in sorted_data:
            entry_date = pd.Timestamp(entry['date'].split()[0]).date()
            if entry_date <= date_list[current_target_idx]:
                #print(f"entry_date ({entry_date}) <= {date_list[current_target_idx]}")
                results[labels[current_target_idx]]=round(entry['sma'],2)
                #print(f"so results[{labels[current_target_idx]}] = {round(entry['sma'],2)}")
                current_target_idx+=1
            if current_target_idx>=len(labels):
                break
    return results

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

def extract_CashAndCash(date,sorted_balance_data):
    if sorted_balance_data is None:
        return None

    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    elif isinstance(date, datetime):
        date = date.date()

    min_date = date - timedelta(days=100)

    for entry in sorted_balance_data:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date() 
        if min_date <= entry_date <= date:
            return round(entry['cashAndCashEquivalents'], 2)
        
    return None

def extract_total_debt(date,sorted_balance_data):
    if sorted_balance_data is None:
        return None

    if isinstance(date, str):
        date = datetime.strptime(date, '%Y-%m-%d').date()
    elif isinstance(date, pd.Timestamp):
        date = date.date()
    elif isinstance(date, datetime):
        date = date.date()

    min_date = date - timedelta(days=100)

    for entry in sorted_balance_data:
        entry_date = datetime.strptime(entry['date'].split()[0], "%Y-%m-%d").date() 
        if min_date <= entry_date <= date:
            lag_days = (date - entry_date).days
            return round(entry['totalDebt'], 2),lag_days
        
    return None

def extract_EBITDA(input_date,sorted_data_income):
    if sorted_data_income is None:
        return None

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
            lag_days = (input_date - entry_date).days
            return round(entry['ebitda'], 2), lag_days
        
    return None


def extract_RDexpenses(date,sorted_data_income):
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
        if min_date <= entry_date <= date :
            return round(entry['researchAndDevelopmentExpenses'], 2)
        
    return None

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

def extract_current_and_future_estim_REV(today_date, sorted_data):
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

    today_date = ensure_datetime(today_date)


    if sorted_data is None:
        return None,None
    

    
    current_estimated_eps = None
    future_estimated_eps = None

    for i, entry in enumerate(sorted_data):
        if entry is not None:
            entry_date = ensure_datetime(entry['date'])

            if entry_date <= today_date:
                if current_estimated_eps is None:
                    current_estimated_eps = entry['estimatedRevenueAvg']
                
                if i > 0:
                    future_estimated_eps = sorted_data[i-1]['estimatedRevenueAvg']
                
                break
    return current_estimated_eps, future_estimated_eps

def extract_current_and_future_estim_eps(today_date, sorted_data):
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

    today_date = ensure_datetime(today_date)


    if sorted_data is None:
        return None,None
    

    
    current_estimated_eps = None
    future_estimated_eps = None

    for i, entry in enumerate(sorted_data):
        if entry is not None:
            entry_date = ensure_datetime(entry['date'])

            if entry_date <= today_date:
                if current_estimated_eps is None:
                    current_estimated_eps = entry['estimatedEpsAvg']
                
                if i > 0:
                    future_estimated_eps = sorted_data[i-1]['estimatedEpsAvg']
                
                break
    return current_estimated_eps, future_estimated_eps

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

def extract_min_price_in_range(start_date, end_date,historical_data,return_date=False):
    #print(f"extract_min_price_in_range({start_date},{end_date})")
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
            #print("  relevant_data is empty")
            return None
        max_price = relevant_data.min()
        max_price_date = relevant_data.idxmin()
        if return_date:
            #print(f"  return_date is true , and minprice = {max_price}, minpricedate = {max_price_date}")
            return max_price,max_price_date
        else:
            return max_price

    return None

def extract_max_price_in_range(start_date, end_date,historical_data,return_date=False):
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
        max_price_date = relevant_data.idxmax()
        if return_date:
            return max_price,max_price_date
        else:
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


def calculate_sector_relatives(df, sector_column):
    """
    Calculate sector-relative metrics (requires a sector column in the data)
    
    Parameters:
    df (pd.DataFrame): DataFrame with stock features
    sector_column (str): Name of the column containing sector information
    
    Returns:
    pd.DataFrame: DataFrame with additional sector-relative features
    """
    df_new = df.copy()
    
    # Metrics to calculate sector-relatives for
    metrics = ['pe','peg', 'markRevRatio', 'fwdPriceTosale_diff','EVEbitdaRatio', '3M_return','6M_return','1W_return','1M_return','2M_return','4M_return','5M_return','1y_return','6M_return','7M_return']
    
    # Calculate sector-relative metrics
    for metric in metrics:
        if metric in df.columns:
            # Calculate sector means
            sector_means = df.groupby(sector_column)[metric].transform('mean')
            # Calculate sector-relative metric
            df_new[f'{metric}_sector_relative'] = df[metric] / sector_means - 1
    
    return df_new
 
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