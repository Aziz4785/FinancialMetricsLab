import random
from datetime import datetime, timedelta
from .utils import *
from ..common import *
import itertools
import pickle
"""
to run : py -m metrics.Metric8.simulation  (if it doesnt work add.py)
to make the most of the grid search
run it several time with not a big number of stocks (250 is good) and pick the 
most consistent combination of parameters
"""

MIN_BUDGET_BY_STOCK = 500

TAKE_PROFIT_DOLLARS = 25
STOP_LOSS_DOLLARS = -20
GOOD_RETURN_DOLLARS = 10
BAD_RETURN_DOLLARS = 0

NBR_OF_SIMULATION = 500

CASHFLOW_DEBT_LOWER_BOUND =0.18 #0.14
EPS_GROWTH_LOWER_BOUND =0.07 #0.06
PS_UPPER_BOUND=6.74 #7.74

WAITING_IN_WEEKS = 5



GRID_SEARCH = False
SAVE_ONLY_BAD_PRED=False
USE_ML = True
DEBUG_FREQUENCY=3
SAMPLE_SIZE = 200
stocks = load_stocks(SAMPLE_SIZE,'stock_list.csv')
stored_mc_dict = load_stored_dict('market_cap_data.json')
stored_totalDebt_dict = load_stored_dict('totalDebt_data.json')
#stocks = ['F','GM','AAL','CAH','MCK','UAL','GOOG','AMZN','NVDA','DELL','HBI','KR']
#historical_data_for_stock,high_df,low_df, market_cap_dict, income_dict, hist_data_df_for_stock,balance_dict,cashflow_dict,estimations_dict ,sector_dict= fetch_stock_data(stocks)
historical_data_for_stock,vwap_dict,volume_dict,open_dict,high_df,low_df, market_cap_dict, income_dict, hist_data_df_for_stock,balance_dict,cashflow_dict,estimations_dict ,sector_dict= fetch_stock_data(stocks)
if USE_ML:
    sma_10d_dict,sma_50w_dict,sma_100d_dict,sma_200d_dict,sma_50d_dict,_,_,_ = fetch_data_for_ml(stocks)
    model,scaler,model2,scaler2,model3,scaler3,model4,scaler4 = load_models()

print("sleep just before simulation")
time.sleep(30)

today = pd.Timestamp.today()

min_random_date = today - pd.DateOffset(months=60)
max_random_date = today - pd.DateOffset(weeks=WAITING_IN_WEEKS+1)

date_range = pd.date_range(start=min_random_date, end=max_random_date, freq='D')
date_range = date_range[(date_range.weekday < 5) & (date_range.year != 2020)]
max_simulation = min(NBR_OF_SIMULATION,len(date_range)-1)
random_dates = random.sample(list(date_range), max_simulation)
if not GRID_SEARCH:
    print("length of random_dates : ",len(random_dates))

def simulation():
    nbr_stored_mc=0
    total_gain=[]
    data_list = []
    nbr_good=0
    nbr_bad=0
    nbr_total=0
    unique_stocks = set()
    if (CASHFLOW_DEBT_LOWER_BOUND<=0.12 and EPS_GROWTH_LOWER_BOUND<=0.09 and PS_UPPER_BOUND>=8.7) or (CASHFLOW_DEBT_LOWER_BOUND<=0.13 and EPS_GROWTH_LOWER_BOUND<=0.07 and PS_UPPER_BOUND>=9):
        print("   ->skipping..")
        return -9999
    print(f"simulation(  CASHFLOW_DEBT_LOWER_BOUND = {CASHFLOW_DEBT_LOWER_BOUND}, EPS_GROWTH_LOWER_BOUND {EPS_GROWTH_LOWER_BOUND} , PS_UPPER_BOUND = {PS_UPPER_BOUND}")
    
    #delta_days = (max_random_date - min_random_date).days
    for i,random_date in enumerate(random_dates):
        random_date = random_date.date()
        gain_on_that_simul=0
        date_after_1W = random_date + pd.DateOffset(weeks=WAITING_IN_WEEKS)
        nbr_of_None_prices = 0
        nbr_of_None_income = 0
        if not GRID_SEARCH and i%DEBUG_FREQUENCY==0:
                print(f"Simulation {i+1}: {random_date}")

        stocks_in_range = []
        for stock in stocks:
            if historical_data_for_stock[stock] is None or hist_data_df_for_stock[stock] is None or income_dict[stock] is None:
                if income_dict[stock] is None:
                    nbr_of_None_income+=1
                # if not GRID_SEARCH:
                #     if i%DEBUG_FREQUENCY==0:
                #         print("one of the input data is None")
                continue
            price_at_date = extract_stock_price_at_date(random_date,historical_data_for_stock[stock])
            if price_at_date is None or price_at_date<=0.1:
                nbr_of_None_prices+=1
                continue
            
            #
            income_features = extract_income_features(random_date,income_dict[stock])


            if stock in stored_mc_dict:
                if random_date in stored_mc_dict[stock]:
                    nbr_stored_mc +=1
                    market_cap = stored_mc_dict[stock][random_date]
                else :
                    market_cap = extract_market_cap(random_date,market_cap_dict[stock])
                    stored_mc_dict[stock][random_date] = market_cap
            else:
                market_cap = extract_market_cap(random_date,market_cap_dict[stock])
                stored_mc_dict[stock] = {}
                stored_mc_dict[stock][random_date] = market_cap

            if stock in stored_totalDebt_dict:
                if random_date in stored_totalDebt_dict[stock]:
                    totalDebt = stored_totalDebt_dict[stock][random_date]
                else :
                    balance_features = extract_balance_features(random_date,balance_dict[stock])
                    totalDebt = safe_dict_get(balance_features,'totalDebt')
                    stored_totalDebt_dict[stock][random_date] = totalDebt
            else:
                balance_features = extract_balance_features(random_date,balance_dict[stock])
                totalDebt = safe_dict_get(balance_features,'totalDebt')
                stored_totalDebt_dict[stock] = {}
                stored_totalDebt_dict[stock][random_date] = totalDebt 

            #ps_ratio = calculate_ps_ratio(market_cap,income_features)
            balance_features = extract_balance_features(random_date,balance_dict[stock])
            estimations_features = extract_estimation_features(random_date,estimations_dict[stock])
            cashflow_features =extract_cashflow_features(random_date,cashflow_dict[stock])
            #pe = calculate_pe_from_features(price_at_date,income_features)
            #free_cf = safe_dict_get(cashflow_features, 'freeCashFlow')
            CashProvidedByOperatingActivities = safe_dict_get(cashflow_features,'netCashProvidedByOperatingActivities')
            #totalDebt = safe_dict_get(balance_features,'totalDebt')
            #eps = safe_dict_get(income_features, 'eps')
            current_estimated_eps = safe_dict_get(estimations_features, 'current_estimated_eps')
            future_eps =safe_dict_get(estimations_features, 'future_estimated_eps')
            #current_est_revenue =safe_dict_get(estimations_features, 'current_estim_rev')
            revenues =safe_dict_get(income_features, 'revenue')
            #total_asset = extract_total_asset_from_features(balance_features)
            #dividend_payout_ratio = calculate_div_payout_ratio(random_date,cashflow_dict[stock],income_dict[stock])
            #ev_ebitda = calculate_evebitda_from_features(random_date,market_cap,balance_features,income_features)
            #ev_ebitda = calculate_evebitda(random_date,market_cap_dict[stock],balance_dict[stock],income_dict[stock])
            #dividendsPaid = calculate_dividendPaid(random_date,cashflow_dict[stock])
            #current_estimated_eps, future_eps = extract_current_and_future_estim_eps(random_date, estimations_dict[stock])
            est_eps_growth = safe_divide(safe_subtract(future_eps,current_estimated_eps),current_estimated_eps)
            ps = safe_divide(market_cap,revenues)
            cashflow_debt = safe_divide(CashProvidedByOperatingActivities,totalDebt)

            #future_eps = calculate_FutureEps(random_date,estimations_dict[stock])
            #if dividend_payout_ratio is None:
                #nbr_of_None_evgp_ratio+=1
            if cashflow_debt is not None and cashflow_debt>=CASHFLOW_DEBT_LOWER_BOUND and ps is not None and ps<=PS_UPPER_BOUND:
                if est_eps_growth is not None and est_eps_growth>=EPS_GROWTH_LOWER_BOUND:
                    if USE_ML:
                        features1=['sma_10d_5months_ago', 'month', 'ebitda', 'cashcasheq', 'netDebt', 'curr_est_eps', 'max_minus_min', 'dist_min8M_4M', 'max_minus_min8M', 'Spread4Mby8M', 'debtToPrice', 'peg', 'PS_to_PEG', 'dividend_payout_ratio', 'EVEbitdaRatio', 'fwdPriceTosale_diff', 'sma_100d_to_sma_200d_ratio', 'combined_valuation_score'] 
                        
                        prediction, buy_probability = predict_buy(model,scaler,features1,random_date,stock,est_eps_growth,price_at_date,
                                                                sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                                income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock[stock],high_df[stock],low_df[stock])

                        features2=['sma_200d', 'sma_10d_5months_ago', 'sma_10d_11months_ago', 'income_lag_days', 'marketCap', 'EV', 'max_minus_min', 'min8M_lag', 'Spread4Mby8M', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'EVRevenues', 'fwdPriceTosale', 'fwdPriceTosale_diff', 'deriv_2w', 'deriv_2m', 'deriv2_1_3m', 'deriv2_2_3m', 'deriv_4m', 'deriv_5m', 'combined_valuation_score', 'sma10_yoy_growth']
                        
                        prediction2, buy_probability2 = predict_buy(model2,scaler2,features2,random_date,stock,est_eps_growth,price_at_date,
                                                                sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                                income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock[stock],high_df[stock],low_df[stock])

                        features3 = ['sma_10d_6months_ago', 'marketCap', 'balance_lag_days', 'max_minus_min', 'min8M_lag', 'Spread4Mby8M', 'markRevRatio', 'peg', 'EVEbitdaRatio', 'EVGP', 'fwdPriceTosale', 'fwdPriceTosale_diff', 'deriv_2w', 'deriv_2m', 'deriv2_1_3m', 'deriv2_2_3m', 'deriv_4m', 'deriv_5m', '1y_return', 'combined_valuation_score']
                        prediction3, buy_probability3 = predict_buy(model3,scaler3,features3,random_date,stock,est_eps_growth,price_at_date,
                                                                sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                                income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock[stock],high_df[stock],low_df[stock])

                        features4 = ['est_eps_growth', 'month', 'GP', 'netIncome', 'ebitda', 'marketCap', 'cashcasheq', 'curr_est_eps', 'max_minus_min', 'dist_max8M_4M', 'dist_min8M_4M', 'ebitdaMargin', 'Spread4Mby8M', 'markRevRatio', 'peg', 'dividend_payout_ratio', 'EVEbitdaRatio', 'EVGP', 'fwdPriceTosale_diff', 'deriv_min8M', '1y_return', '3M_2M_growth', 'price_to_sma_100d_ratio', 'sma_10d_to_sma_50d_ratio', 'sma_100d_to_sma_200d_ratio', 'combined_valuation_score']
                        prediction4, buy_probability4 = predict_buy(model4,scaler4,features4,random_date,stock,est_eps_growth,price_at_date,
                                                                sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                                income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock[stock],high_df[stock],low_df[stock])

                        print(f"stock {stock}  prediction : {prediction}   prediction2 = {prediction2} prediction3 = {prediction3}")
                        if prediction ==1 and buy_probability>0.82 and prediction2==1 and prediction3==1 and prediction4==1:
                            if not GRID_SEARCH and i%DEBUG_FREQUENCY==0:
                                print(f"we will buy {stock} (prob1 = {buy_probability})  (prob2 = {buy_probability2}) prob3({buy_probability3})  prob4({buy_probability4})")
                            stocks_in_range.append((stock, price_at_date,est_eps_growth))
                    else:
                        stocks_in_range.append((stock, price_at_date,est_eps_growth))
                

        #if not GRID_SEARCH:
                #if i%DEBUG_FREQUENCY==0:
                    #print(f"  at that date, fraction of stocks having None incomes : {nbr_of_None_income/len(stocks)}")
                    #if (len(stocks)-nbr_of_None_income)!=0:
                       # print(f"                fraction of valid stocks having None prices : {nbr_of_None_prices/(len(stocks)-nbr_of_None_income)}")
                        #print(f"                fraction of valid stocks having None pr ratio: {nbr_of_None_evgp_ratio/(len(stocks)-nbr_of_None_income)}")
        if len(stocks_in_range)<=0:
            continue

        stocks_to_buy = stocks_in_range
        if not GRID_SEARCH and i%DEBUG_FREQUENCY==0:
                print("stocks to buy : ",stocks_to_buy)
        
        for stock_to_buy in stocks_to_buy:
            buy = True
            
            stock = stock_to_buy[0]
            initial_price = stock_to_buy[1]

            quantity = int(MIN_BUDGET_BY_STOCK/initial_price)
            gain_for_this_stock=0
            if buy and quantity>0:
                target_price = TAKE_PROFIT_DOLLARS/quantity + initial_price +0.01
                stop_loss_price = initial_price + STOP_LOSS_DOLLARS/quantity - 0.01

                _,take_profit_date = find_first_price_threshold(random_date, date_after_1W, high_df[stock], target_price,'profit', return_date=True)
                _,stop_loss_date = find_first_price_threshold(random_date, date_after_1W, low_df[stock], stop_loss_price,'loss', return_date=True)
        
                #max_price_1W = extract_max_price_in_range(random_date,date_after_1W,high_df[stock])
                #min_price_1W = extract_min_price_in_range(random_date,date_after_1W,low_df[stock])
                
                #if max_price_1W is None or initial_price is None or min_price_1W is None:
                    #continue
                price_after_period = extract_stock_price_at_date(date_after_1W,historical_data_for_stock[stock])
                if price_after_period is None or price_after_period <=0.1:
                    continue
                #max_gain_for_this_stock = (max_price_1W - initial_price)*quantity
                #min_gain_for_this_stock = (min_price_1W - initial_price)*quantity
                gain_for_this_stock_after_period = (price_after_period - initial_price)*quantity
                if not GRID_SEARCH and i%DEBUG_FREQUENCY==0:
                        print(f"we buy {quantity} of {stock} at {initial_price}$ it achieves tp at {take_profit_date} and it achieves sl at {stop_loss_date}")
                
                to_buy = None
                if stop_loss_date is not None and take_profit_date is not None and take_profit_date<stop_loss_date:
                    to_buy=1
                    if not GRID_SEARCH and i%DEBUG_FREQUENCY==0:
                        print(f"  we make {TAKE_PROFIT_DOLLARS} from that trade")
                    gain_on_that_simul +=TAKE_PROFIT_DOLLARS
                elif stop_loss_date is not None and take_profit_date is not None and take_profit_date>stop_loss_date:
                    to_buy=0
                    if not GRID_SEARCH and i%DEBUG_FREQUENCY==0:
                        print(f"  we make {STOP_LOSS_DOLLARS} from that trade")
                    gain_on_that_simul +=STOP_LOSS_DOLLARS
                elif stop_loss_date is None and take_profit_date is not None :
                    to_buy=1
                    gain_on_that_simul +=TAKE_PROFIT_DOLLARS
                elif stop_loss_date is not None and take_profit_date is None :
                    to_buy=0
                    gain_on_that_simul +=STOP_LOSS_DOLLARS
                elif gain_for_this_stock_after_period >= GOOD_RETURN_DOLLARS:
                    to_buy=1
                    gain_on_that_simul +=gain_for_this_stock_after_period
                    if not GRID_SEARCH and i%DEBUG_FREQUENCY==0:
                        print(f"  we make {gain_for_this_stock_after_period} from that trade")
                elif gain_for_this_stock_after_period<=BAD_RETURN_DOLLARS:
                    to_buy=0
                    gain_on_that_simul +=gain_for_this_stock_after_period
                    if not GRID_SEARCH and i%DEBUG_FREQUENCY==0:
                        print(f"  we make {gain_for_this_stock_after_period} from that trade")

                if to_buy is not None:
                    if to_buy==0:
                        nbr_bad+=1
                        
                    elif to_buy ==1:
                        nbr_good+=1

                    unique_stocks.add(stock)
                    if not GRID_SEARCH:
                        data_list.append({
                            'date': random_date,
                            'symbol': stock,
                            'price': initial_price,
                            'est_eps_growth':stock_to_buy[2],
                            'to_buy': to_buy
                        })

                    nbr_total+=1
                

        
        if (gain_on_that_simul!=0):
            if not GRID_SEARCH:
                if i%DEBUG_FREQUENCY==0:
                    print(f"  ->total result : {gain_on_that_simul} ")
            total_gain.append(gain_on_that_simul)
    if not GRID_SEARCH:
        print("nbr good : ",nbr_good)
        print("nbr_bad : ",nbr_bad)
        training_data = pd.DataFrame(data_list)
        if 'date' in training_data.columns:
            training_data['date'] = pd.to_datetime(training_data['date'])

        
        csv_filename = 'training_data.csv'
        print(f"unique stocks  : {len(unique_stocks)}")
        if NBR_OF_SIMULATION>=100:
            training_data.to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")
        return total_gain
    else:
        total_gain = np.array(total_gain)
        mean = np.mean(total_gain)
        min_stocks,max_stocks = unique_stocks_threshold(SAMPLE_SIZE,NBR_OF_SIMULATION)
        if nbr_total>0 and len(unique_stocks)>min_stocks and len(unique_stocks)<max_stocks: #this is hardocded if we sample 250 stocks for the simulation
            score = (nbr_good-nbr_bad)/((nbr_total)*WAITING_IN_WEEKS)
            if score >-10:
                print(f"   ->score : {score}  mean : {mean}  uniquestocks = {len(unique_stocks)}")
        else:
            if len(unique_stocks)<=min_stocks:
                print("too few stocks ",len(unique_stocks))
            if len(unique_stocks)>=max_stocks:
                print("too many stocks ",len(unique_stocks))
            score=-99999
        return score

if not GRID_SEARCH:
    total_gain = simulation()
    #print(total_gain)
    total_gain = np.array(total_gain)
    mean = np.mean(total_gain)
    print("mean : ",mean)
    print(total_gain)

def grid_search_simulation():
    # Define the ranges for each parameter
    param1_range = [0.14,0.15,0.16,0.18] 
    param2_range =[0.04,0.05,0.06,0.07,0.08,0.09,0.1]
    param3_range = [6.74,7.74]
    param4_range = [5]

    parameter_combinations = list(itertools.product(param1_range,param2_range,param3_range,param4_range))

    best_result = float('-inf')
    best_params = None

    for params in parameter_combinations:
        param1 , param2, param3,param4= params
        

        global CASHFLOW_DEBT_LOWER_BOUND,EPS_GROWTH_LOWER_BOUND, PS_UPPER_BOUND,WAITING_IN_WEEKS
        CASHFLOW_DEBT_LOWER_BOUND =param1
        EPS_GROWTH_LOWER_BOUND =param2
        PS_UPPER_BOUND=param3
        WAITING_IN_WEEKS = param4

        result = simulation()

        # Update the best result if the current result is better
        if result > best_result:
            best_result = result
            best_params = params

    return best_params, best_result


if GRID_SEARCH:
    

    print("grid search")
    best_parameters, best_score = grid_search_simulation()

    print(f"Best parameters: "
        f"CASHFLOW_DEBT_LOWER_BOUND={best_parameters[0]:.2f}",
        f"EPS_GROWTH_LOWER_BOUND={best_parameters[1]:.2f}",
        f"PS_UPPER_BOUND={best_parameters[2]:.2f}",
        f"WAITING_IN_WEEKS={best_parameters[3]:.2f}")
    print(f"Best score: {best_score}")

filepath = save_market_cap_dict(stored_mc_dict, save_format='json')
print(f"marketcap Data saved to: {filepath}")