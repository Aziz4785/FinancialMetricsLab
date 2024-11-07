import pandas as pd
from .utils import *



def get_metric1_stocks():
    """
    Reads stock data and calculates Metric1 for each stock.
    Returns a list of dictionaries containing symbol, metric value, and buy signal.
    """
    RATIO_UPPER_BOUND = 0.8 
    SELL_LIMIT_PERCENTAGE_1 = 0.13 
    try:
        today = pd.Timestamp.today()
        # Read the common stock data file
        df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/stock_list.csv')

        model,scaler,model2,scaler2,model3,scaler3,model4,scaler4 = load_models()

        # Get unique symbols
        symbols = df['Ticker'].unique()
        
        results = []
        for stock in symbols:
            current_price = get_current_price(stock)
            if current_price is None:
                continue
                
            #peg = calculate_current_quarter_peg(stock, current_price)
            peg = calculate_quarter_peg_ratio(stock, today.date(), current_price, prefetched_data=None)
            should_buy = 0
            target = None
            if peg is not None and peg <= RATIO_UPPER_BOUND and peg>=0:
                print(f"stock {stock} , peg : {peg}")
                market_cap = get_current_market_cap(stock)
                income_dict = get_income_statements(stock, 'quarter')

                features_for_pred =['MarkRevRatio', 'sma_200d', 'price', 'sma_50w']
                prediction, buy_probability = predict_buy(model,scaler,today,stock,peg,features_for_pred,current_price,None, None, None, None, income_dict,market_cap_dict=market_cap)
                #print("prediction 1 : ",prediction)
                prediction2, buy_probability = predict_buy(model2,scaler2,today,stock,peg,['MarkRevRatio', 'sma_50w', 'sma_200d', 'sma_100d', 'price', 'sma_10d'],current_price,None, None, None, None, income_dict,market_cap_dict=market_cap)
                #print("prediction2  : ",prediction2)
                prediction3, buy_probability = predict_buy(model3,scaler3,today,stock,peg,['MarkRevRatio', 'sma_200d', 'price', 'sma_50w', 'sma_100d', 'ratio'],current_price,None, None, None, None, income_dict,market_cap_dict=market_cap)
                #print("prediction3  : ",prediction3)
                prediction4, buy_probability = predict_buy(model4,scaler4,today,stock,peg,['MarkRevRatio', 'sma_200d', 'price', 'sma_50w', 'sma_100d', 'ratio', 'ebitdaMargin'],current_price,None, None, None, None, income_dict,market_cap_dict=market_cap)
                
                
                # Set shouldBuy based on predictions
                if prediction == 1 and prediction2 == 1 and prediction3==1 and prediction4==1:
                    should_buy = 3
                    if current_price is not None:
                        target = round(current_price*(1+SELL_LIMIT_PERCENTAGE_1),2)
                elif prediction is None and prediction2 is None and prediction3 is None and prediction4 is None :
                    should_buy = 2
                else:
                    should_buy = 1

            
            if peg is not None:
                peg = round(peg,2)
            
            # Add to results
            results.append({
                "symbol": stock,
                "metric_1": peg,
                "shouldBuy_1": should_buy,
                "target_1":target,
            })
        
        return results
    
    except Exception as e:
        print(f"Error in get_metric1_stocks: {e}")
        return []