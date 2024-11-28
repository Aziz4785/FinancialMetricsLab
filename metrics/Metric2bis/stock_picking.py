import pandas as pd
from .utils import *

"""
to run : py -m metrics.Metric2bis.stock_picking 
"""
PS_UPPER_BOUND = 1.4
PS_LOWER_BOUND = 0
def get_Metric2bis_stocks():
    """
    Reads stock data and calculates Metric2bis for each stock.
    Returns a list of dictionaries containing symbol, metric value, and buy signal.
    """
    print("---get_Metric2bis_stocks()-----")
    #try:
    today = pd.Timestamp.today()
    # Read the common stock data file
    df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/stock_list.csv')
    
    model,scaler,model2,scaler2,model3 = load_models()

    # Get unique symbols
    symbols = df['Ticker'].unique()
    results = []
    for i,stock in enumerate(symbols):
        if i%25==0:
            print(i)
        current_price = get_current_price(stock)
        if current_price is None:
            continue

        income_statement = get_income_dict(stock,'quarter')
        estimation_statement = get_estimation_dict(stock,'quarter')
    
        income_features = extract_income_features(today.date(),income_statement)
        market_cap = get_current_market_cap(stock)
        balance_features = extract_balance_features(today.date(),get_balance_shit(stock,'quarter'))
        estimations_features = extract_estimation_features(today.date(),estimation_statement)
        cashflow_features =extract_cashflow_features(today.date(),get_cashflow_dict(stock,'quarter'))

        #ev_ebitda = calculate_evebitda_from_features(today.date(),market_cap,balance_features,income_features)
        #current_estimated_eps, future_eps = extract_current_and_future_estim_eps(today.date(), estimation_statement)
        ps_ratio = calculate_ps_ratio(market_cap,income_features)

        should_buy = 0
        target = None
        if ps_ratio is not None and ps_ratio<=PS_UPPER_BOUND and ps_ratio>=PS_LOWER_BOUND:
            print(f"{stock} is eligible")
            features1=['revenues', 'price', 'pe', 'ps_ratio', 'sma_100d', 'sma_200d']
            features2=['price', 'sma_100d', 'sma_50w', 'sma_200d', 'sma_50d', 'GP', 'dividendsPaid', 'marketCap', 'EV', 'max_in_1M', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'EVRevenues', 'fwdPriceTosale', 'price_to_sma_200d_ratio', 'combined_valuation_score']
            
            
            sma_10d_dict = get_SMA(stock, '1day', 10)
            sma_50w_dict = get_SMA(stock,  '1week', 50)
            sma_100d_dict = get_SMA(stock, '1day', 100)
            sma_200d_dict = get_SMA(stock, '1day', 200)
            sma_50d_dict = get_SMA(stock, '1day', 50)
            
            hist_data_df_for_stock = convert_to_df(convert_to_dict(get_historical_price(stock)))

            prediction, buy_probability = predict_buy(model,scaler,features1,today.date(),stock,ps_ratio,current_price,
                                                        sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                        income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock)
            
            prediction2, buy_probability2 = predict_buy(model2,scaler2,features2,today.date(),stock,ps_ratio,current_price,
                                                        sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                        income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock)

            prediction3, buy_probability3 = predict_buy(model3,scaler2,features2,today.date(),stock,ps_ratio,current_price,
                                                        sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                        income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock)

            print(f"PRED : {prediction} ({buy_probability})  PRED2 : {prediction2} ({buy_probability2}) PRED3 : {prediction3} ({buy_probability3})")
            print("possible target : ",round(current_price*1.11,2))
            # Set shouldBuy based on predictions
            if prediction ==1 and  buy_probability>=0.98 and prediction2 ==1 and buy_probability2>=0.98 and prediction3==1 and buy_probability3>=0.98:
                should_buy = 3
                if current_price is not None:
                    target = round(current_price*1.11,2)
            elif prediction is None:
                should_buy = 2
            else:
                should_buy = 1

        
        if ps_ratio is not None:
            ps_ratio = round(ps_ratio,2)
        
        """print({
            "symbol": stock,
            "metric_5": ps_ratio,
            "shouldBuy_5": should_buy,
            "target_5":target,
        })"""
        # Add to results
        results.append({
            "symbol": stock,
            "metric_5": ps_ratio,
            "shouldBuy_5": should_buy,
            "target_5":target,
        })

    return results
    
    # except Exception as e:
    #     print(f"Error in get_Metric2bis_stocks: {e}")
    #     return []
    
get_Metric2bis_stocks()