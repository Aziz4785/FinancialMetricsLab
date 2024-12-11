import pandas as pd
from .utils import *

"""
to run : py -m metrics.Metric7.stock_picking 
"""
MIN_BUDGET_BY_STOCK = 300
TAKE_PROFIT_DOLLARS = 25
STOP_LOSS_DOLLARS = -20
FREE_CF_LOWER_BOUND =282972500
EPS_GROWTH_LOWER_BOUND =0.16 
PE_UPPER_BOUND=51.51
EST_REV_LOWER_BOUND =8100127380

def get_Metric7_stocks():
    """
    Reads stock data and calculates Metric7 for each stock.
    Returns a list of dictionaries containing symbol, metric value, and buy signal.
    """
    print("---get_Metric7_stocks()-----")
    #try:
    today = pd.Timestamp.today()
    # Read the common stock data file
    df = pd.read_csv('stock_list.csv')
    #df = df.tail(300)
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
        #ps_ratio = calculate_ps_ratio(market_cap,income_features)
        free_cf = safe_dict_get(cashflow_features, 'freeCashFlow')
        eps = safe_dict_get(income_features, 'eps')
        current_estimated_eps = safe_dict_get(estimations_features, 'current_estimated_eps')
        future_eps =safe_dict_get(estimations_features, 'future_estimated_eps')
        current_est_revenue =safe_dict_get(estimations_features, 'current_estim_rev')
        est_eps_growth = safe_divide(safe_subtract(future_eps,current_estimated_eps),current_estimated_eps)
        peRatio = safe_divide(current_price,eps)
        
        should_buy = 0
        target = None
        if free_cf is not None and free_cf>=FREE_CF_LOWER_BOUND and peRatio is not None and peRatio<=PE_UPPER_BOUND:
            if est_eps_growth is not None and est_eps_growth>=EPS_GROWTH_LOWER_BOUND and current_est_revenue is not None and current_est_revenue>=EST_REV_LOWER_BOUND:
                print(f"{stock} is eligible")
                
                sma_10d_dict = get_SMA(stock, '1day', 10)
                sma_50w_dict = get_SMA(stock,  '1week', 50)
                sma_100d_dict = get_SMA(stock, '1day', 100)
                sma_200d_dict = get_SMA(stock, '1day', 200)
                sma_50d_dict = get_SMA(stock, '1day', 50)
                
                historic_data = get_historical_price(stock)

                #hist_data_df_for_stock = convert_to_df(convert_to_dict(historic_data))
                date_to_close,high_dict,low_dict = convert_to_dicts(historic_data,parse_time=True)

                hist_data_df_for_stock= convert_to_df(date_to_close,'close')

                high_df = convert_to_df(high_dict,'high')
                low_df= convert_to_df(low_dict,'low')
                #print(high_df)
                features1=['price', 'est_eps_growth', 'sma_100d', 'sma_200d', 'month', 'GP', 'netIncome', 'marketCap', 'cashcasheq', 'EV', 'dist_max8M_4M', 'markRevRatio', 'netDebtToPrice', 'dividend_payout_ratio', 'EVGP', 'fwdPriceTosale_diff', 'deriv_6m', 'deriv_min8M']
                            
                prediction, buy_probability = predict_buy(model,scaler,features1,today.date(),stock,est_eps_growth,current_price,
                                                        sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                        income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock,high_df,low_df)

                features2=['price', 'sma_100d', 'sma_200d', 'sma_50d', 'sma_10d_4months_ago', 'sma_10d_6months_ago', 'marketCap', 'EV', 'max_minus_min8M', 'debtToPrice', 'pe', 'peg', 'netDebtToPrice', 'dividend_payout_ratio', 'EVEbitdaRatio', 'EVGP', 'deriv_4m', 'deriv_5m', 'deriv_6m', 'sma_50d_to_sma_200d_ratio', 'combined_valuation_score', 'sma10_yoy_growth']
                
                prediction2, buy_probability2 = predict_buy(model2,scaler2,features2,today.date(),stock,est_eps_growth,current_price,
                                                        sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                        income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock,high_df,low_df)
                prediction3, buy_probability3 = predict_buy(model3,scaler2,features2,today.date(),stock,est_eps_growth,current_price,
                                                        sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                        income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock,high_df,low_df)


                print(f"PRED : {prediction} ({buy_probability})  PRED2 : {prediction2} ({buy_probability2})  PRED3 : {prediction3} ({buy_probability3})")
                quantity = int(MIN_BUDGET_BY_STOCK/current_price)
                if quantity is not None and quantity>0:
                    print(f"price target : {current_price}  -> {current_price+round(TAKE_PROFIT_DOLLARS/quantity,2)}")
                    print(f"stop loss : {current_price}  -> {current_price+round(STOP_LOSS_DOLLARS/quantity,2)}")
                    # Set shouldBuy based on predictions
                    if prediction ==1 and  buy_probability>=0.8 and prediction2 ==1 and buy_probability2>=0.8 and prediction3 ==1 and buy_probability3>=0.8:
                        should_buy = 3
                        if current_price is not None:
                            target = round(current_price*1.103,2)
                    elif prediction is None:
                        should_buy = 2
                    else:
                        should_buy = 1

        
        if est_eps_growth is not None:
            est_eps_growth = round(est_eps_growth,2)
        
        """print({
            "symbol": stock,
            "metric_5": pe,
            "shouldBuy_5": should_buy,
            "target_5":target,
        })"""
        # Add to results
        results.append({
            "symbol": stock,
            "metric_5": est_eps_growth,
            "shouldBuy_5": should_buy,
            "target_5":target,
        })

    return results
    
    # except Exception as e:
    #     print(f"Error in get_Metric7_stocks: {e}")
    #     return []
    
get_Metric7_stocks()