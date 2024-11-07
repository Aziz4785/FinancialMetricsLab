import pandas as pd
from .utils import *

"""
to run : py -m metrics.Metric2.stock_picking 
"""

def get_metric2_stocks():
    """
    Reads stock data and calculates Metric2 for each stock.
    Returns a list of dictionaries containing symbol, metric value, and buy signal.
    """
    print("---get_metric2_stocks()-----")
    try:
        today = pd.Timestamp.today()
        # Read the common stock data file
        df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/stock_list.csv')
        
        model,scaler,model2 = load_models()

        # Get unique symbols
        symbols = df['Ticker'].unique()
        
        results = []
        for i,stock in enumerate(symbols):
            if i%50==0:
                print(i)
            current_price = get_current_price(stock)
            if current_price is None:
                continue

            income_statement = get_income_statements(stock, 'quarter')
            pr_ratio = calculate_current_price_to_revenue_ratio(stock,today.date(),None,income_statement,do_api_call=True)

            should_buy = 0
            target = None
            if pr_ratio is not None and pr_ratio <= 1.2 and pr_ratio>=0:
                features_for_pred = ['revenues', 'price', 'pe_ratio', 'ebitdaMargin', 'ratio', 'sma_100d', 'sma_200d']
                prediction, buy_probability = predict_buy(model,scaler,today,stock,features_for_pred,current_price,pr_ratio,None,None,None ,None ,income_statement)

                features_for_pred = ['revenues', 'price', 'pe_ratio', 'ebitdaMargin', 'ratio', 'sma_100d', 'sma_200d']
                prediction2, buy_probability2 = predict_buy(model2,scaler,today,stock,features_for_pred,current_price,pr_ratio,None,None,None ,None ,income_statement)
                
                # Set shouldBuy based on predictions
                if prediction == 1 and prediction2 == 1:
                    should_buy = 3
                    if current_price is not None:
                        target = round(current_price*1.14,2)
                elif prediction is None and prediction2 is None:
                    should_buy = 2
                else:
                    should_buy = 1

            
            if pr_ratio is not None:
                pr_ratio = round(pr_ratio,2)
            
            # Add to results
            results.append({
                "symbol": stock,
                "metric_2": pr_ratio,
                "shouldBuy_2": should_buy,
                "target_2":target,
            })

        return results
    
    except Exception as e:
        print(f"Error in get_metric2_stocks: {e}")
        return []
    
#get_metric2_stocks()