import random
from datetime import datetime, timedelta
from .utils import *
from ..common import *
import itertools
import pickle
"""
to run : py -m metrics.Metric2bis.simulation  (if it doesnt work add.py)
to make the most of the grid search
run it several time with not a big number of stocks (250 is good) and pick the 
most consistent combination of parameters
"""

MIN_BUDGET_BY_STOCK = 1000
TRANSACITON_FEE = 1.5
PS_UPPER_BOUND = 1.4
PS_LOWER_BOUND = 0
NBR_OF_SIMULATION = 1000
SELL_LIMIT_PERCENTAGE_1 = 11
WAITING_IN_WEEKS = 14
A_GOOD_RETURN = 0.1
A_BAD_RETURN = 0.05
GRID_SEARCH = False
SAVE_ONLY_BAD_PRED=True
USE_ML = True
DEBUG_FREQUENCY=2
stocks = load_stocks(300,'C:/Users/aziz8/Documents/FinancialMetricsLab/stock_list.csv')
#stocks = ['F','GM','AAL','CAH','MCK','UAL','GOOG','AMZN','NVDA','DELL','HBI','KR']
historical_data_for_stock, market_cap_dict, income_dict, hist_data_df_for_stock,balance_dict,cashflow_dict,estimations_dict,_ = fetch_stock_data(stocks)

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
    print(f"simulation( SELL_LIMIT_PERCENTAGE_1 = {SELL_LIMIT_PERCENTAGE_1} , PS_UPPER_BOUND = {PS_UPPER_BOUND} ")
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
            if price_at_date is None:
                nbr_of_None_prices+=1
                continue
            
            #
            income_features = extract_income_features(random_date,income_dict[stock])
            market_cap = extract_market_cap(random_date,market_cap_dict[stock])
            ps_ratio = calculate_ps_ratio(market_cap,income_features)
            balance_features = extract_balance_features(random_date,balance_dict[stock])
            estimations_features = extract_estimation_features(random_date,estimations_dict[stock])
            cashflow_features =extract_cashflow_features(random_date,cashflow_dict[stock])
            #total_asset = extract_total_asset_from_features(balance_features)
            #dividend_payout_ratio = calculate_div_payout_ratio(random_date,cashflow_dict[stock],income_dict[stock])
            #ev_ebitda = calculate_evebitda_from_features(random_date,market_cap,balance_features,income_features)
            #ev_ebitda = calculate_evebitda(random_date,market_cap_dict[stock],balance_dict[stock],income_dict[stock])
            #revenue_est_growth = calculate_revenue_growth(random_date,estimations_dict[stock])
            #dividendsPaid = calculate_dividendPaid(random_date,cashflow_dict[stock])
            #current_estimated_eps, future_eps = extract_current_and_future_estim_eps(random_date, estimations_dict[stock])
            #future_eps = calculate_FutureEps(random_date,estimations_dict[stock])
            #if dividend_payout_ratio is None:
                #nbr_of_None_evgp_ratio+=1
            if ps_ratio is not None and ps_ratio<=PS_UPPER_BOUND and ps_ratio>=PS_LOWER_BOUND:
                if USE_ML:
                    features1=['revenues', 'price', 'pe', 'ps_ratio', 'sma_100d', 'sma_200d']
                    
                    prediction, buy_probability = predict_buy(model,scaler,features1,random_date,stock,ps_ratio,price_at_date,
                                                              sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                              income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock[stock])

                    features2=['price', 'sma_100d', 'sma_50w', 'sma_200d', 'sma_50d', 'GP', 'dividendsPaid', 'marketCap', 'EV', 'max_in_1M', 'pe', 'peg', 'EVEbitdaRatio', 'EVGP', 'EVRevenues', 'fwdPriceTosale', 'price_to_sma_200d_ratio', 'combined_valuation_score']
                    
                    prediction2, buy_probability2 = predict_buy(model2,scaler2,features2,random_date,stock,ps_ratio,price_at_date,
                                                              sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                              income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock[stock])

                    prediction3, buy_probability3 = predict_buy(model3,scaler2,features2,random_date,stock,ps_ratio,price_at_date,
                                                              sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                              income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock[stock])

                    #print(f"stock {stock}  prediction : {prediction}  ")
                    if prediction ==1 and  buy_probability>=0.98 and prediction2 ==1 and buy_probability2>=0.98 and prediction3==1 and buy_probability3>=0.98:
                        if not GRID_SEARCH and i%DEBUG_FREQUENCY==0:
                            print(f"we will buy {stock} (prob1 = {buy_probability})  (prob2 = {buy_probability2}) (prob3 = {buy_probability3})")
                        stocks_in_range.append((stock, price_at_date,ps_ratio))
                else:
                    stocks_in_range.append((stock, price_at_date,ps_ratio))
                

        #if not GRID_SEARCH:
                #if i%DEBUG_FREQUENCY==0:
                    #print(f"  at that date, fraction of stocks having None incomes : {nbr_of_None_income/len(stocks)}")
                    #if (len(stocks)-nbr_of_None_income)!=0:
                       # print(f"                fraction of valid stocks having None prices : {nbr_of_None_prices/(len(stocks)-nbr_of_None_income)}")
                        #print(f"                fraction of valid stocks having None pr ratio: {nbr_of_None_evgp_ratio/(len(stocks)-nbr_of_None_income)}")
        if len(stocks_in_range)<=0:
            continue
        #stocks_in_range.sort(key=lambda x: x[1])

        #budget_per_stock = INTITIAL_CAPTIAL/len(stocks_in_range)
        budget_per_stock = MIN_BUDGET_BY_STOCK

        stocks_to_buy = stocks_in_range
        # if not GRID_SEARCH:
        #     if i%DEBUG_FREQUENCY==0:
        #         print("stocks to buy : ",stocks_to_buy)
        for stock_to_buy in stocks_to_buy:
            buy = True
            
            stock = stock_to_buy[0]
            initial_price = stock_to_buy[1]

            quantity = int(budget_per_stock/initial_price)
            gain_for_this_stock=0
            if buy:
                max_price_1W = extract_max_price_in_range(random_date,date_after_1W,hist_data_df_for_stock[stock])
                if max_price_1W is None or initial_price is None:
                    continue
                max_percentage_increase = (max_price_1W - initial_price) / initial_price * 100
                price_after_1W = extract_stock_price_at_date(date_after_1W,historical_data_for_stock[stock])
                if price_after_1W is None:
                    continue
                if not GRID_SEARCH:
                    if i%DEBUG_FREQUENCY==0:
                        print(f"we buy {quantity} of {stock} at {initial_price}$ its max price is {max_price_1W} and after {WAITING_IN_WEEKS} week (on {date_after_1W}) it is {price_after_1W}")
                if max_percentage_increase >= SELL_LIMIT_PERCENTAGE_1:
                    gain_for_this_stock= (SELL_LIMIT_PERCENTAGE_1/100)*initial_price*quantity - 2*TRANSACITON_FEE
                else:
                    gain_for_this_stock = (price_after_1W - initial_price)*quantity - 2*TRANSACITON_FEE
                if not GRID_SEARCH:
                    if i%DEBUG_FREQUENCY==0:
                        print(f"   -> {gain_for_this_stock} $ (good if it is above {quantity*initial_price*A_GOOD_RETURN})")
                if gain_for_this_stock >= quantity*initial_price*A_GOOD_RETURN:
                    nbr_good +=1
                    unique_stocks.add(stock)
                    if not GRID_SEARCH and not SAVE_ONLY_BAD_PRED:
                        data_list.append({
                            'date': random_date,
                            'symbol': stock,
                            'price': initial_price,
                            'ps_ratio':stock_to_buy[2],
                            'to_buy': 1
                        })
                elif gain_for_this_stock < quantity*initial_price*A_BAD_RETURN:
                    unique_stocks.add(stock)
                    nbr_bad+=1
                    if not GRID_SEARCH:
                        data_list.append({
                            'date': random_date,
                            'symbol': stock,
                            'price': initial_price,
                            'ps_ratio':stock_to_buy[2],
                            'to_buy': 0
                        })
                nbr_total+=1
                gain_on_that_simul+=gain_for_this_stock    

        
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
        if nbr_total>0 and len(unique_stocks)>13 and len(unique_stocks)<30: #this is hardocded if we sample 250 stocks for the simulation
            score = (nbr_good-nbr_bad)/((nbr_total)*WAITING_IN_WEEKS)
            if score >-10:
                print(f"   ->score : {score}  mean : {mean}  uniquestocks = {len(unique_stocks)}")
        else:
            if len(unique_stocks)<=12:
                print("too few stocks ",len(unique_stocks))
            if len(unique_stocks)>=30:
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
    param1_range = [11,11.5] 
    param2_range = [1.2,1.4,1.6,2]
    param3_range = [14]

    parameter_combinations = list(itertools.product(param1_range,param2_range,param3_range))

    best_result = float('-inf')
    best_params = None

    for params in parameter_combinations:
        param1 , param2, param3= params
        

        global SELL_LIMIT_PERCENTAGE_1,PS_UPPER_BOUND, WAITING_IN_WEEKS
        SELL_LIMIT_PERCENTAGE_1 = param1
        PS_UPPER_BOUND = param2
        WAITING_IN_WEEKS = param3

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
        f"SELL_LIMIT_PERCENTAGE_1={best_parameters[0]:.2f}, "
        f"PS_UPPER_BOUND={best_parameters[1]:.2f}",
        f"WAITING_IN_WEEKS={best_parameters[2]:.2f}")
    print(f"Best score: {best_score}")

