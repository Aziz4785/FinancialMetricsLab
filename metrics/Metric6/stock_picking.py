import pandas as pd
from .utils import *

"""
to run : py -m metrics.Metric6.stock_picking 
"""

# FSLY is eligible
# PRED : 1 (0.89)  PRED2 : 1 (0.925)

SELL_LIMIT_PERCENTAGE_1 = 10.3
PE_UPPER_BOUND = 20
PE_LOWER_BOUND = -40
GP_UPPER_BOUND = 222791500
def get_Metric6_stocks():
    """
    Reads stock data and calculates Metric6 for each stock.
    Returns a list of dictionaries containing symbol, metric value, and buy signal.
    """
    print("---get_Metric6_stocks()-----")
    #try:
    today = pd.Timestamp.today()
    # Read the common stock data file
    df = pd.read_csv('C:/Users/aziz8/Documents/FinancialMetricsLab/stock_list.csv')
    
    model,scaler,model2,scaler2 = load_models()

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
        #ps_ratio = calculate_ps_ratio(market_cap,income_features)
        pe = calculate_pe_from_features(current_price,income_features)
        gp = safe_dict_get(income_features, 'grossProfit')
        
        should_buy = 0
        target = None
        if pe is not None and pe<=PE_UPPER_BOUND and pe>=PE_LOWER_BOUND and gp is not None and gp<=GP_UPPER_BOUND:
            print(f"{stock} is eligible")
            
            sma_10d_dict = get_SMA(stock, '1day', 10)
            sma_50w_dict = get_SMA(stock,  '1week', 50)
            sma_100d_dict = get_SMA(stock, '1day', 100)
            sma_200d_dict = get_SMA(stock, '1day', 200)
            sma_50d_dict = get_SMA(stock, '1day', 50)
            
            hist_data_df_for_stock = convert_to_df(convert_to_dict(get_historical_price(stock)))

            features1=['price', 'sma_100d', 'RnD_expenses', 'EV', 'maxPercen_4M', 'max_in_2W', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'fwdPriceTosale', 'fwdPriceTosale_diff', '3M_2M_growth', '1Y_6M_growth', 'combined_valuation_score', 'sma10_yoy_growth']
                        
            prediction, buy_probability = predict_buy(model,scaler,features1,today.date(),stock,pe,current_price,
                                                    sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                    income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock)

            features2=['price', 'month', 'RnD_expenses', 'revenues', 'GP', 'dividendsPaid', 'curr_est_eps', 'EV', 'min_in_4M', 'max_minus_min', 'max_in_8M', 'dist_min8M_4M', 'ebitdaMargin', 'eps_growth', 'peg', 'PS_to_PEG', 'fwdPriceTosale_diff', 'var_sma50D_100D', '1Y_6M_growth', 'combined_valuation_score']
            
            prediction2, buy_probability2 = predict_buy(model2,scaler2,features2,today.date(),stock,pe,current_price,
                                                    sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                    income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock)


            print(f"PRED : {prediction} ({buy_probability})  PRED2 : {prediction2} ({buy_probability2})")
            print(f"possible target : {current_price}  -> {round(current_price*1.103,2)}")
            # Set shouldBuy based on predictions
            if prediction ==1 and  buy_probability>=0.98 and prediction2 ==1 and buy_probability2>=0.98:
                should_buy = 3
                if current_price is not None:
                    target = round(current_price*1.103,2)
            elif prediction is None:
                should_buy = 2
            else:
                should_buy = 1

        
        if pe is not None:
            pe = round(pe,2)
        
        """print({
            "symbol": stock,
            "metric_5": pe,
            "shouldBuy_5": should_buy,
            "target_5":target,
        })"""
        # Add to results
        results.append({
            "symbol": stock,
            "metric_5": pe,
            "shouldBuy_5": should_buy,
            "target_5":target,
        })

    return results
    
    # except Exception as e:
    #     print(f"Error in get_Metric6_stocks: {e}")
    #     return []
    
get_Metric6_stocks()