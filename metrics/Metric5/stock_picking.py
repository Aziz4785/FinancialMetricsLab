import pandas as pd
from .utils import *

"""
to run : py -m metrics.Metric5.stock_picking 
"""

def get_Metric5_stocks():
    """
    Reads stock data and calculates Metric5 for each stock.
    Returns a list of dictionaries containing symbol, metric value, and buy signal.
    """
    print("---get_Metric5_stocks()-----")
    #try:
    today = pd.Timestamp.today()
    # Read the common stock data file
    df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/stock_list.csv')
    #df = df.tail(300)
    model,scaler = load_models()

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

        ev_ebitda = calculate_evebitda_from_features(today.date(),market_cap,balance_features,income_features)
        current_estimated_eps, future_eps = extract_current_and_future_estim_eps(today.date(), estimation_statement)


        should_buy = 0
        target = None
        if ev_ebitda is not None and ev_ebitda <= 30 and ev_ebitda>=-30 and future_eps is not None and future_eps<=0.01:
            print(f"{stock} is eligible")
            features_for_pred = ['evebitda', 'sma_200d', 'sma_50d', 'marketCap', 'markRevRatio', 'EVRevenues', 'fwdPriceTosale', 'deriv_2m', '1Y_6M_growth', 'sma_100d_to_sma_200d_ratio', 'combined_valuation_score', 'sma10_yoy_growth']
            sma_10d_dict = get_SMA(stock, '1day', 10)
            sma_50w_dict = get_SMA(stock,  '1week', 50)
            sma_100d_dict = get_SMA(stock, '1day', 100)
            sma_200d_dict = get_SMA(stock, '1day', 200)
            sma_50d_dict = get_SMA(stock, '1day', 50)
            hist_data_df_for_stock = convert_to_df(convert_to_dict(get_historical_price(stock)))
            prediction, buy_probability = predict_buy(model,scaler,features_for_pred,today.date(),stock,ev_ebitda,current_price,
                                                                sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                                income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock)

            
            print(f"preddiction : {prediction} with probab = {buy_probability}")
            print(f"possible target : {current_price}  -> {round(current_price*1.109,2)}")
            # Set shouldBuy based on predictions
            if prediction == 1 and buy_probability >= 0.97:
                should_buy = 3
                if current_price is not None:
                    target = round(current_price*1.109,2)
            elif prediction is None:
                should_buy = 2
            else:
                should_buy = 1

        
        if ev_ebitda is not None:
            ev_ebitda = round(ev_ebitda,2)
        
        """print({
            "symbol": stock,
            "metric_5": ev_ebitda,
            "shouldBuy_5": should_buy,
            "target_5":target,
        })"""
        # Add to results
        results.append({
            "symbol": stock,
            "metric_5": ev_ebitda,
            "shouldBuy_5": should_buy,
            "target_5":target,
        })

    return results
    
    # except Exception as e:
    #     print(f"Error in get_Metric5_stocks: {e}")
    #     return []
    
get_Metric5_stocks()