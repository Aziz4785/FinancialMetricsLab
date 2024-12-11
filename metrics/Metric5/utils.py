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
from ..common import * 
load_dotenv()

# Access the API key
api_key = os.getenv('FMP_API_KEY')


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
        historic_data = get_historical_price(stock)
        date_to_close = convert_to_dict(historic_data)
        hist_data_df_for_stock[stock] = convert_to_df(date_to_close)
        historical_data_for_stock[stock] = date_to_close
        #sector_dict[stock]=get_company_sector(stock)
        market_cap_dict[stock] = get_historical_market_cap(stock)
        revenues_dict[stock] = get_income_dict(stock,'quarter')
        balance_shit_dict[stock] = get_balance_shit(stock,'quarter')
        cashflow_dict[stock]= get_cashflow_dict(stock,'quarter')
        estimations_dict[stock] = get_estimation_dict(stock,'quarter')
        if estimations_dict[stock] is None:
            print("estimations dict is none for : ",stock)
        if c % 40==0 and c > 0:
            print(c)
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
    available_metrics = [metric for metric in valuation_metrics if metric in df_new.columns]

    if available_metrics:
        df_new['combined_valuation_score'] = df_new[available_metrics].mean(axis=1)

    # 4. Size and Value Combinations
    # Market cap quintile (1 = smallest, 5 = largest)
    if 'marketCap' in df_new:
        df_new['market_cap_quintile'] = pd.qcut(df_new['marketCap'], q=5, labels=[1,2,3,4,5], duplicates='drop')
    
    # Value score combining different metrics (handling missing values)
    value_metrics = []
    
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
    
    if ev_metrics:
        df_new['ev_composite'] = df_new[ev_metrics].mean(axis=1)

    
    return df_new


def load_models():
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric5/models/XGB7_11_12.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
    with open('C:/Users/aziz8/Documents/FinancialMetricsLab/metrics/Metric5/models/scaler_11_12.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    return model,scaler

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
        sma_50w_dict[stock]=get_SMA(stock,  '1week', 50)
        sma_100d_dict[stock]=get_SMA(stock, '1day', 100)
        sma_200d_dict[stock]=get_SMA(stock, '1day', 200)
        sma_50d_dict[stock]=get_SMA(stock, '1day', 50)
        #market_cap_dict[stock] = get_historical_market_cap(stock)
        #balance_dict[stock] = get_balance_shit(stock,'quarter')
        #std_10d_dict[stock] = get_std(stock,'1day',10)
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


def predict_buy(model,scaler,features_for_pred,input_date,symbol,ev_ebitda,price_at_date,
                                                                  sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                                  income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock):
                
                   
    max_in_8M, max8M_date = extract_max_price_in_range((pd.to_datetime(input_date) - pd.DateOffset(months=8)).date(),input_date,hist_data_df_for_stock,return_date=True)
    min_in_8M, min8M_date = extract_min_price_in_range((pd.to_datetime(input_date) - pd.DateOffset(months=8)).date(),input_date,hist_data_df_for_stock,return_date=True)
    pe = safe_divide(price_at_date,safe_dict_get(estimations_features,'current_estimated_eps'))
    eps_growth = safe_divide((safe_dict_get(estimations_features,'future_estimated_eps')-safe_dict_get(estimations_features,'current_estimated_eps')),safe_dict_get(estimations_features,'current_estimated_eps'))
    peg = safe_divide(pe,eps_growth)
    EV = calculate_EV(market_cap,balance_features,income_features)
    evgp = safe_divide(EV,safe_dict_get(income_features, 'grossProfit'))
    markRevRAtio  = safe_divide(market_cap,safe_dict_get(income_features,'revenue'))
    if(isinstance(sma_100d_dict, list)):
        sma100d = extract_SMA(input_date,sma_100d_dict)
        sma50d = extract_SMA(input_date,sma_50d_dict)
        sma10d = extract_SMA(input_date,sma_10d_dict)
        sma_50w = extract_SMA(input_date,sma_50w_dict)
        sma_200d = extract_SMA(input_date,sma_200d_dict)
        sma_10d_11mago = extract_SMA((pd.to_datetime(input_date) - pd.DateOffset(months=11)).date(), sma_10d_dict)
        sma_10d_6mago = extract_SMA((pd.to_datetime(input_date) - pd.DateOffset(months=6)).date(), sma_10d_dict)
        sma_10d_2months_ago = extract_SMA((pd.to_datetime(input_date) - pd.DateOffset(months=2)).date(), sma_10d_dict)
    else:
        if sma_100d_dict is None:
            sma100d = None
        else:
            sma100d = extract_SMA(input_date,sma_100d_dict[symbol])
        sma50d = extract_SMA(input_date,sma_50d_dict[symbol])
        sma10d = extract_SMA(input_date,sma_10d_dict[symbol])
        sma_50w = extract_SMA(input_date,sma_50w_dict[symbol])
        sma_200d = extract_SMA(input_date,sma_200d_dict[symbol])
        sma_10d_11mago = extract_SMA((pd.to_datetime(input_date) - pd.DateOffset(months=11)).date(), sma_10d_dict[symbol])
        sma_10d_6mago = extract_SMA((pd.to_datetime(input_date) - pd.DateOffset(months=6)).date(), sma_10d_dict[symbol])
        sma_10d_2months_ago = extract_SMA((pd.to_datetime(input_date) - pd.DateOffset(months=2)).date(), sma_10d_dict[symbol])
    #[  'max_in_2W',  'fwdPriceTosale_diff']
    feature_calculations = {
        'evebitda': lambda: ev_ebitda,
        'netIncome': lambda: safe_dict_get(income_features,'netIncome'),
        'sma_50w': lambda: sma_50w,
        'sma_50d': lambda: sma50d,
        'sma_200d': lambda: sma_200d,
        'sma_100d': lambda: sma100d,
        'sma_10d': lambda: sma10d,
        'price': lambda: price_at_date,
        'marketCap': lambda: market_cap,
        'EV': lambda: EV,
        'sma10_yoy_growth': lambda: safe_multiply(safe_subtract(safe_divide(sma10d,sma_10d_11mago),1), 100),
        'deriv_2m': lambda:safe_divide(safe_subtract(price_at_date,sma_10d_2months_ago),2),
        'sma_100d_to_sma_200d_ratio': lambda : safe_subtract(safe_divide(sma100d, sma_200d), 1),
        'max_in_8M': lambda: max_in_8M,
        'max_minus_min8M': lambda:safe_subtract(max_in_8M,min_in_8M),
        'markRevRatio': lambda: markRevRAtio ,
        'eps_growth': lambda: eps_growth,
        'pe': lambda: pe,
        'peg': lambda: peg,
        'EVGP': lambda: evgp,
        'combined_valuation_score' :lambda: safe_divide(safe_add(safe_add(safe_add(pe, peg),evgp),markRevRAtio),4),
        'EVRevenues': lambda: safe_divide(EV,safe_dict_get(income_features,'revenue')),
        'fwdPriceTosale': lambda: safe_divide(market_cap,safe_dict_get(estimations_features,'future_estim_rev')),
        'var_sma100D': lambda:safe_divide(safe_subtract(price_at_date,sma100d),sma100d),
        'var_sma50D_100D': lambda:safe_divide(safe_subtract(sma50d,sma100d),sma100d),
        'var_sma10D_100D':  lambda:safe_divide(safe_subtract(sma10d,sma100d),sma100d),
        '1y_return' :lambda: safe_divide(safe_subtract(price_at_date,sma_10d_11mago),sma_10d_11mago),
        '1Y_6M_growth': lambda: safe_divide(safe_subtract(sma_10d_6mago,sma_10d_11mago),sma_10d_11mago),
    }

    # Calculate only the required features
    data_dict = {'date': input_date, 'symbol': symbol}
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
    probability = model.predict_proba(X_scaled)[0, 1]  # Probability of class 1 (buy)
    #probability=1
    return prediction[0], probability


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

def calculate_div_payout_ratio(input_date,cashflow_data,income_data):
    div_paid = extract_dividend_paid(input_date,cashflow_data)
    net_income = extract_net_inome(input_date,income_data)

    if div_paid is None or net_income is None or net_income ==0:
        return None 
    return div_paid/net_income

def calculate_dividendPaid(input_date,cashflow_dict):
    dividendsPaid = extract_dividend_paid(input_date,cashflow_dict)
    return dividendsPaid

def calculate_evebitda(random_date,market_cap_data,balance_data,income_data):
    mc = extract_market_cap(random_date,market_cap_data)
    #total_debt = extract_total_debt(random_date,balance_data)
    #cashncasheq = extract_Field(random_date, balance_data,'cashAndCashEquivalents')
    balance_features = extract_balance_features(random_date,balance_data)
    income_features= extract_income_features(random_date,income_data)
    if balance_features is None or income_features is None:
        return None
    total_debt=balance_features['totalDebt']
    cashncasheq = balance_features['cashAndCashEquivalents']
    ebitda = income_features['ebitda']
    if ebitda is None or mc is None or total_debt is None or ebitda==0:
        return None
    EV =  mc+total_debt-cashncasheq
    return EV/ebitda



def calculate_evebitda_from_features(random_date,mc,balance_features,income_features):
    if income_features is None or balance_features is None:
        return None
    ebitda = income_features['ebitda']
    EV =  calculate_EV(mc,balance_features,income_features)
    if EV==0:
        return None
    return safe_divide(EV,ebitda)

def calculate_pe_ratio(input_date,price,income_data):
    eps_income = extract_net_inome(input_date,income_data)

    if eps_income is None or price is None or eps_income ==0:
        return None 
    return price/eps_income

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

    if return_date == True:
        return None,None
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
    metrics = ['pe','peg', 'EVGP','EVRevenues','markRevRatio', 'fwdPriceTosale_diff','EVEbitdaRatio', '3M_return','6M_return','1W_return','1M_return','2M_return','4M_return','5M_return','1y_return','6M_return','7M_return','max_minus_min', 'deriv_3m', 'deriv_4m',]
    
    # Calculate sector-relative metrics
    for metric in metrics:
        if metric in df.columns:
            # Calculate sector means
            sector_means = df.groupby(sector_column)[metric].transform('mean')
            # Calculate sector-relative metric
            df_new[f'{metric}_sector_relative'] = df[metric] / sector_means - 1
    
    return df_new

