import pandas as pd
from .utils import *

"""
THIS DOESNT GIVE GOOD RESULTS :
CONCLUSION : EV/GP sucks !
"""
def get_metric3_stocks():
    """
    Reads stock data and calculates Metric3 for each stock.
    Returns a list of dictionaries containing symbol, metric value, and buy signal.
    """
    try:
        # Read the common stock data file
        df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/stock_list.csv')
        
        # Get unique symbols
        symbols = df['Ticker'].unique()
        
        results = []
        for symbol in symbols:
            # Get data for this symbol
            stock_data = df[df['Ticker'] == symbol].copy()
            
            # Calculate metric3 for this stock
            metric_value = 3.89
            
            # Determine if we should buy
            should_buy = 2
            
            # Add to results
            results.append({
                "symbol": symbol,
                "metric_3": metric_value,
                "shouldBuy_3": should_buy
            })
        
        return results
    
    except Exception as e:
        print(f"Error in get_metric3_stocks: {e}")
        return []