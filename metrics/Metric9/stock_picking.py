import pandas as pd
from .utils import *

"""
to run : py -m metrics.Metric9.stock_picking 
"""
MIN_BUDGET_BY_STOCK = 400

TAKE_PROFIT_DOLLARS =40
STOP_LOSS_DOLLARS = -20
GOOD_RETURN_DOLLARS = 10
BAD_RETURN_DOLLARS = 0

DIV_PAYOUT_RATIO_LOWER = 0
GROSSPROFIT_UPPER = 878957000
EPS_LOWER = 0.05 #or 0.08
GPRATIO_LOWER =0.44

WAITING_IN_WEEKS = 5


def get_Metric9_stocks():
    """
    Reads stock data and calculates Metric9 for each stock.
    Returns a list of dictionaries containing symbol, metric value, and buy signal.
    """
    print("---get_Metric9_stocks()-----")
    #try:
    today = pd.Timestamp.today()
    # Read the common stock data file
    df = pd.read_csv('stock_list.csv')
    #df = df.tail(300)
    model,scaler,model2,scaler2,model3,scaler3 = load_models()

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
        #free_cf = safe_dict_get(cashflow_features, 'freeCashFlow')
        CashProvidedByOperatingActivities = safe_dict_get(cashflow_features,'netCashProvidedByOperatingActivities')
        #eps = safe_dict_get(income_features, 'eps')
        totalDebt = safe_dict_get(balance_features,'totalDebt')
        current_estimated_eps = safe_dict_get(estimations_features, 'current_estimated_eps')
        future_eps =safe_dict_get(estimations_features, 'future_estimated_eps')
        #current_est_revenue =safe_dict_get(estimations_features, 'current_estim_rev')
        est_eps_growth = safe_divide(safe_subtract(future_eps,current_estimated_eps),current_estimated_eps)
        revenues =safe_dict_get(income_features, 'revenue')
        eps = safe_dict_get(income_features, 'eps')
        gross_profit = safe_dict_get(income_features,'grossProfit')
        #est_eps_growth = safe_divide(safe_subtract(future_eps,current_estimated_eps),current_estimated_eps)
        ps = safe_divide(market_cap,revenues)
        cashflow_debt = safe_divide(CashProvidedByOperatingActivities,totalDebt)
        dividend_payout_ratio = calculate_div_payout_ratio(today.date(),cashflow_features,income_features)
        rnd = safe_dict_get(income_features,'researchAndDevelopmentExpenses')
        free_cashflow = safe_dict_get(cashflow_features,'freeCashFlow')
        gp_ratio = calculate_growthpotential_ratio(
            rnd,
            free_cashflow,
            revenues
        )
        #ev_ebitda

        should_buy = 0
        
        target = None
        if dividend_payout_ratio is not None and dividend_payout_ratio>=DIV_PAYOUT_RATIO_LOWER and gross_profit is not None and gross_profit<=GROSSPROFIT_UPPER:
            if eps is not None and eps>=EPS_LOWER and gp_ratio is not None and gp_ratio>= GPRATIO_LOWER:
                print(f"{stock} is eligible")
                
                sma_10d_dict = get_SMA(stock, '1day', 10)
                sma_50w_dict = get_SMA(stock,  '1week', 50)
                sma_100d_dict = get_SMA(stock, '1day', 100)
                sma_200d_dict = {}#get_SMA(stock, '1day', 200)
                sma_50d_dict = {}#get_SMA(stock, '1day', 50)
                
                historic_data = get_historical_price(stock)
                date_to_close,high_dict,low_dict,open_dict,vwap_dict,volume_dict  = convert_to_dicts(historic_data,parse_time=True,with_vwap_volume=False)
                hist_data_df_for_stock= convert_to_df(date_to_close,'close')
                high_df = convert_to_df(high_dict,'high')
                low_df= convert_to_df(low_dict,'low')


                features1=['sma_10d', 'sma_10d_4months_ago', 'sma_10d_11months_ago', 'marketCap', 'markRevRatio', 'EVEbitdaRatio']
                        
                prediction, buy_probability = predict_buy(model,scaler,features1,today.date(),stock,gp_ratio,current_price,
                                                        sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                        income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock,high_df,low_df)

                features2=['price', 'sma_50w', 'sma_100d', 'peg', 'EVRevenues', 'combined_valuation_score']
                prediction2, buy_probability2 = predict_buy(model2,scaler2,features2,today.date(),stock,gp_ratio,current_price,
                                                        sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                        income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock,high_df,low_df)

                features3 = ['sma_10d', 'sma_10d_4months_ago', 'sma_10d_11months_ago', 'month', 'netIncome', 'marketCap', 'max_minus_min', 'min8M_lag', 'markRevRatio', 'pe', 'EVEbitdaRatio', 'fwdPriceTosale', 'fwdPriceTosale_diff', 'deriv_2m', 'deriv_3m', 'deriv2_1_3m', 'deriv_4m', 'deriv_5m', 'deriv_6m', '7M_return', '1y_return', '1Y_6M_growth']
                prediction3, buy_probability3 = predict_buy(model3,scaler3,features3,today.date(),stock,gp_ratio,current_price,
                                                        sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                        income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock,high_df,low_df)



                print(f"PRED : {prediction} ({buy_probability})  PRED2 : {prediction2} ({buy_probability2})  PRED3 : {prediction3} ({buy_probability3}) ")
                quantity = int(MIN_BUDGET_BY_STOCK/current_price)
                if quantity is not None and quantity>0:
                    target_price = current_price+round(TAKE_PROFIT_DOLLARS/quantity,2)
                    print(f" (quantity : {quantity}) , price target : {current_price}  -> {target_price}")
                    print(f"stop loss : {current_price}  -> {current_price+round(STOP_LOSS_DOLLARS/quantity,2)}")
                    # Set shouldBuy based on predictions
                    if prediction ==1 and buy_probability>0.86 and prediction2==1 and prediction3==1:
                        print("ITS A BUY")
                        should_buy = 3
                        if current_price is not None:
                            target = round(target_price,2)
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
    #     print(f"Error in get_Metric9_stocks: {e}")
    #     return []
    
get_Metric9_stocks()