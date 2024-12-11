import random
from datetime import datetime, timedelta
from .utils import *
from ..common import *
import itertools
import pickle
"""
to run : py -m metrics.Metric7.simulation  (if it doesnt work add.py)
to make the most of the grid search
run it several time with not a big number of stocks (250 is good) and pick the 
most consistent combination of parameters
"""

MIN_BUDGET_BY_STOCK = 300

TAKE_PROFIT_DOLLARS = 25
STOP_LOSS_DOLLARS = -20
GOOD_RETURN_DOLLARS = 5
BAD_RETURN_DOLLARS = -5

NBR_OF_SIMULATION = 500

FREE_CF_LOWER_BOUND =282972500
EPS_GROWTH_LOWER_BOUND =0.16 
PE_UPPER_BOUND=51.51
EST_REV_LOWER_BOUND =8100127380

WAITING_IN_WEEKS = 10



GRID_SEARCH = False
SAVE_ONLY_BAD_PRED=False
USE_ML = True
DEBUG_FREQUENCY=2
SAMPLE_SIZE = 350
stocks = load_stocks(SAMPLE_SIZE,'stock_list.csv')
#stocks = ['F','GM','AAL','CAH','MCK','UAL','GOOG','AMZN','NVDA','DELL','HBI','KR']
#historical_data_for_stock, market_cap_dict, income_dict, hist_data_df_for_stock,balance_dict,cashflow_dict,estimations_dict,_ = fetch_stock_data(stocks)
historical_data_for_stock,high_df,low_df, market_cap_dict, income_dict, hist_data_df_for_stock,balance_dict,cashflow_dict,estimations_dict ,sector_dict= fetch_stock_data(stocks)
if USE_ML:
    sma_10d_dict,sma_50w_dict,sma_100d_dict,sma_200d_dict,sma_50d_dict,_,_,_ = fetch_data_for_ml(stocks)
    model,scaler,model2,scaler2,model3 = load_models()

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
    total_gain=[]
    data_list = []
    nbr_good=0
    nbr_bad=0
    nbr_total=0
    unique_stocks = set()
    print(f"simulation(  FREE_CF_LOWER_BOUND = {FREE_CF_LOWER_BOUND}, EPS_GROWTH_LOWER_BOUND {EPS_GROWTH_LOWER_BOUND} , PE_UPPER_BOUND = {PE_UPPER_BOUND} , EST_REV_LOWER_BOUND = {EST_REV_LOWER_BOUND}")
    #delta_days = (max_random_date - min_random_date).days
    for i,random_date in enumerate(random_dates):
        random_date = random_date.date()
        gain_on_that_simul=0
        date_after_1W = random_date + pd.DateOffset(weeks=WAITING_IN_WEEKS)
        nbr_of_None_prices = 0
        nbr_of_None_income = 0
        if not GRID_SEARCH:
            if i%DEBUG_FREQUENCY==0:
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
            market_cap = extract_market_cap(random_date,market_cap_dict[stock])
            #ps_ratio = calculate_ps_ratio(market_cap,income_features)
            balance_features = extract_balance_features(random_date,balance_dict[stock])
            estimations_features = extract_estimation_features(random_date,estimations_dict[stock])
            cashflow_features =extract_cashflow_features(random_date,cashflow_dict[stock])
            #pe = calculate_pe_from_features(price_at_date,income_features)
            free_cf = safe_dict_get(cashflow_features, 'freeCashFlow')
            eps = safe_dict_get(income_features, 'eps')
            current_estimated_eps = safe_dict_get(estimations_features, 'current_estimated_eps')
            future_eps =safe_dict_get(estimations_features, 'future_estimated_eps')
            current_est_revenue =safe_dict_get(estimations_features, 'current_estim_rev')
            #total_asset = extract_total_asset_from_features(balance_features)
            #dividend_payout_ratio = calculate_div_payout_ratio(random_date,cashflow_dict[stock],income_dict[stock])
            #ev_ebitda = calculate_evebitda_from_features(random_date,market_cap,balance_features,income_features)
            #ev_ebitda = calculate_evebitda(random_date,market_cap_dict[stock],balance_dict[stock],income_dict[stock])
            #dividendsPaid = calculate_dividendPaid(random_date,cashflow_dict[stock])
            #current_estimated_eps, future_eps = extract_current_and_future_estim_eps(random_date, estimations_dict[stock])
            est_eps_growth = safe_divide(safe_subtract(future_eps,current_estimated_eps),current_estimated_eps)
            peRatio = safe_divide(price_at_date,eps)
            #future_eps = calculate_FutureEps(random_date,estimations_dict[stock])
            #if dividend_payout_ratio is None:
                #nbr_of_None_evgp_ratio+=1
            if free_cf is not None and free_cf>=FREE_CF_LOWER_BOUND and peRatio is not None and peRatio<=PE_UPPER_BOUND:
                if est_eps_growth is not None and est_eps_growth>=EPS_GROWTH_LOWER_BOUND and current_est_revenue is not None and current_est_revenue>=EST_REV_LOWER_BOUND:
                    if USE_ML:
                        features1=['price', 'est_eps_growth', 'sma_100d', 'sma_200d', 'month', 'GP', 'netIncome', 'marketCap', 'cashcasheq', 'EV', 'dist_max8M_4M', 'markRevRatio', 'netDebtToPrice', 'dividend_payout_ratio', 'EVGP', 'fwdPriceTosale_diff', 'deriv_6m', 'deriv_min8M']
                        
                        prediction, buy_probability = predict_buy(model,scaler,features1,random_date,stock,est_eps_growth,price_at_date,
                                                                sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                                income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock[stock],high_df[stock],low_df[stock])

                        features2=['price', 'sma_100d', 'sma_200d', 'sma_50d', 'sma_10d_4months_ago', 'sma_10d_6months_ago', 'marketCap', 'EV', 'max_minus_min8M', 'debtToPrice', 'pe', 'peg', 'netDebtToPrice', 'dividend_payout_ratio', 'EVEbitdaRatio', 'EVGP', 'deriv_4m', 'deriv_5m', 'deriv_6m', 'sma_50d_to_sma_200d_ratio', 'combined_valuation_score', 'sma10_yoy_growth']
                        
                        prediction2, buy_probability2 = predict_buy(model2,scaler2,features2,random_date,stock,est_eps_growth,price_at_date,
                                                                sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                                income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock[stock],high_df[stock],low_df[stock])

                        prediction3, buy_probability3 = predict_buy(model3,scaler2,features2,random_date,stock,est_eps_growth,price_at_date,
                                                                sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                                income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock[stock],high_df[stock],low_df[stock])


                        print(f"stock {stock}  prediction : {prediction}   prediction2 = {prediction2} prediction3 = {prediction3}")
                        if prediction ==1 and  buy_probability>=0.8 and prediction2 ==1 and buy_probability2>=0.8 and prediction3==1 and buy_probability3>=0.8:
                            if not GRID_SEARCH and i%DEBUG_FREQUENCY==0:
                                print(f"we will buy {stock} (prob1 = {buy_probability})  (prob2 = {buy_probability2}) ")
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
        # if not GRID_SEARCH:
        #     if i%DEBUG_FREQUENCY==0:
        #         print("stocks to buy : ",stocks_to_buy)
        for stock_to_buy in stocks_to_buy:
            buy = True
            
            stock = stock_to_buy[0]
            initial_price = stock_to_buy[1]

            quantity = int(MIN_BUDGET_BY_STOCK/initial_price)
            gain_for_this_stock=0
            if buy:
                max_price_1W = extract_max_price_in_range(random_date,date_after_1W,high_df[stock])
                min_price_1W = extract_min_price_in_range(random_date,date_after_1W,low_df[stock])
                
                if max_price_1W is None or initial_price is None or min_price_1W is None:
                    continue
                price_after_period = extract_stock_price_at_date(date_after_1W,historical_data_for_stock[stock])
                if price_after_period is None or price_after_period <=0.1:
                    continue
                max_gain_for_this_stock = (max_price_1W - initial_price)*quantity
                min_gain_for_this_stock = (min_price_1W - initial_price)*quantity
                gain_for_this_stock_after_period = (price_after_period - initial_price)*quantity
                if not GRID_SEARCH:
                    if i%DEBUG_FREQUENCY==0:
                        print(f"we buy {quantity} of {stock} at {initial_price}$ its max price is {max_price_1W}")
                
                if min_gain_for_this_stock <=STOP_LOSS_DOLLARS:
                    unique_stocks.add(stock)
                    nbr_bad+=1
                    if not GRID_SEARCH:
                        data_list.append({
                            'date': random_date,
                            'symbol': stock,
                            'price': initial_price,
                            'est_eps_growth':stock_to_buy[2],
                            'to_buy': 0
                        })
                    if not GRID_SEARCH and i%DEBUG_FREQUENCY==0:
                        print(f"  -> {STOP_LOSS_DOLLARS} $")
                    gain_on_that_simul+=STOP_LOSS_DOLLARS    
                elif max_gain_for_this_stock >= TAKE_PROFIT_DOLLARS:
                    nbr_good +=1
                    unique_stocks.add(stock)
                    if not GRID_SEARCH and not SAVE_ONLY_BAD_PRED:
                        data_list.append({
                            'date': random_date,
                            'symbol': stock,
                            'price': initial_price,
                            'est_eps_growth':stock_to_buy[2],
                            'to_buy': 1
                        })
                    if not GRID_SEARCH and i%DEBUG_FREQUENCY==0:
                        print(f"  -> {TAKE_PROFIT_DOLLARS} $")
                    gain_on_that_simul+=TAKE_PROFIT_DOLLARS   
                else:
                    if gain_for_this_stock_after_period<=BAD_RETURN_DOLLARS:
                        nbr_bad+=1
                        if not GRID_SEARCH:
                            data_list.append({
                                'date': random_date,
                                'symbol': stock,
                                'price': initial_price,
                                'est_eps_growth':stock_to_buy[2],
                                'to_buy': 0
                            })
                    elif gain_for_this_stock_after_period>=GOOD_RETURN_DOLLARS:
                        nbr_good+=1
                    gain_on_that_simul+=gain_for_this_stock_after_period
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
        max_stocks=60
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
    param1_range = [202972500,282972500,707000000,886000000] 
    param2_range =[0.11,0.13,0.16,0.25]
    param3_range = [45,51.51]
    param4_range = [4875682597,8100127380]
    param5_range = [10]

    parameter_combinations = list(itertools.product(param1_range,param2_range,param3_range,param4_range,param5_range))

    best_result = float('-inf')
    best_params = None

    for params in parameter_combinations:
        param1 , param2, param3,param4, param5= params
        

        global FREE_CF_LOWER_BOUND,EPS_GROWTH_LOWER_BOUND, PE_UPPER_BOUND,EST_REV_LOWER_BOUND,WAITING_IN_WEEKS
        FREE_CF_LOWER_BOUND = param1
        EPS_GROWTH_LOWER_BOUND = param2
        PE_UPPER_BOUND = param3
        EST_REV_LOWER_BOUND = param4
        WAITING_IN_WEEKS = param5

        result = simulation()

        # Update the best result if the current result is better
        if result > best_result:
            best_result = result
            best_params = params

    return best_params, best_result

# # Run the grid search
if GRID_SEARCH:
    print("grid search")
    best_parameters, best_score = grid_search_simulation()

    print(f"Best parameters: "
        f"FREE_CF_LOWER_BOUND={best_parameters[0]:.2f}",
        f"EPS_GROWTH_LOWER_BOUND={best_parameters[1]:.2f}",
        f"PE_UPPER_BOUND={best_parameters[2]:.2f}",
        f"EST_REV_LOWER_BOUND={best_parameters[3]:.2f}",
        f"WAITING_IN_WEEKS={best_parameters[4]:.2f}")
    print(f"Best score: {best_score}")
"""
Best parameters: FREE_CF_LOWER_BOUND=1900000000.00 EPS_GROWTH_LOWER_BOUND=0.20 WAITING_IN_WEEKS=10.00
Best score: -0.0025367156208277704
"""
