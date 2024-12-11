import random
from datetime import datetime, timedelta
from .utils import *
import itertools
import pickle

"""
to run : py -m metrics.Metric2.simulation  (if it doesnt work add.py)
"""

MIN_BUDGET_BY_STOCK = 300
TRANSACITON_FEE = 1.5
RATIO_LOWER_BOUND = 0
RATIO_UPPER_BOUND = 1.2 #1
NBR_OF_SIMULATION = 1000
SELL_LIMIT_PERCENTAGE_1 = 14 #12
WAITING_IN_WEEKS = 16
A_GOOD_RETURN = 0.1
A_BAD_RETURN = 0.05 
GRID_SEARCH = False
DEBUG_FREQUENCY = 5
USE_ML = True
stocks = load_stocks(250,'C:/Users/aziz8/Documents/FinancialMetricsLab/stock_list.csv')
historical_data_for_stock={}
keymetrics_for_stock={}
hist_data_df_for_stock = {}
sma_10d_dict = {}
sma_50w_dict = {}
sma_100d_dict = {}
sma_200d_dict = {}
income_dict = {}
sma_50d_dict = {}
market_cap_dict={}
revenues_dict={}
historical_data_for_stock={}
historical_data_for_stock, market_cap_dict, revenues_dict, hist_data_df_for_stock = fetch_stock_data(stocks)

if USE_ML:
    sma_10d_dict,sma_50w_dict,sma_100d_dict,sma_200d_dict,sma_50d_dict,income_dict = fetch_data_for_ml(stocks)
    model,scaler,model2 = load_models()
    

print("sleep just before simulation")
time.sleep(30)

today = pd.Timestamp.today()
total_gain=[]
min_random_date = today - pd.DateOffset(months=70)
max_random_date = today - pd.DateOffset(months=5)

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
    print(f"simulation( SELL_LIMIT_PERCENTAGE_1 = {SELL_LIMIT_PERCENTAGE_1} , WAITING_IN_WEEKS = {WAITING_IN_WEEKS} , RATIO_UPPER_BOUND = {RATIO_UPPER_BOUND}")
    #delta_days = (max_random_date - min_random_date).days
    for i,random_date in enumerate(random_dates):
        random_date = random_date.date()
        gain_on_that_simul=0
        date_after_1W = random_date + pd.DateOffset(weeks=WAITING_IN_WEEKS)
        nbr_of_None_prices = 0
        nbr_of_None_evgp_ratio = 0
        nbr_of_None_income = 0
        if not GRID_SEARCH:
            if i%DEBUG_FREQUENCY==0:
                print(f"Simulation {i+1}: {random_date}")

        stocks_in_range = []
        for stock in stocks:
            if historical_data_for_stock[stock] is None or hist_data_df_for_stock[stock] is None or income_dict[stock] is None:
                continue
            price_at_date = extract_stock_price_at_date(random_date,historical_data_for_stock[stock])
            if price_at_date is None:
                nbr_of_None_prices+=1
                continue
            
            #dividend_payout_ratio = calculate_div_payout_ratio(random_date,cashflow_dict[stock],income_dict[stock])
            #pe_ratio = calculate_pe_ratio(random_date,price_at_date,income_dict[stock])
            income_features = extract_income_features(random_date,income_dict[stock])
            market_cap = extract_market_cap(random_date,market_cap_dict[stock])
            balance_features = extract_balance_features(random_date,balance_dict[stock])
            estimations_features = extract_estimation_features(random_date,estimations_dict[stock])
            cashflow_features =extract_cashflow_features(random_date,cashflow_dict[stock])

            pr_ratio = calculate_historical_price_to_revenue_ratio(random_date,market_cap,income_features)
            #ev_ebitda = calculate_evebitda(random_date,market_cap_dict[stock],balance_dict[stock],income_dict[stock])
            #dividendsPaid = calculate_dividendPaid(random_date,cashflow_dict[stock])
            current_estimated_eps, future_eps = extract_current_and_future_estim_eps(random_date, estimations_dict[stock])
            #future_eps = calculate_FutureEps(random_date,estimations_dict[stock])
            #if dividend_payout_ratio is None:
                #nbr_of_None_evgp_ratio+=1
            if ev_ebitda is not None and ev_ebitda>=EV_EBITDA_LOW and ev_ebitda<=EV_EBITDA_UP:
                if future_eps is not None and future_eps<=FUTURE_EPS_UP :
                    if USE_ML:
                        features1=['evebitda', 'sma_10d', 'sma_50d', 'marketCap', 'EV', 'max_minus_min8M', 'markRevRatio', 
                         'peg', 'EVGP', 'EVRevenues', 'fwdPriceTosale', 'var_sma100D', 'var_sma50D_100D', 
                         'var_sma10D_100D', '1y_return', '1Y_6M_growth']
                        
                        prediction, buy_probability = predict_buy(model,scaler,features1,random_date,stock,ev_ebitda,price_at_date,
                                                                  sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,
                                                                  income_features,market_cap,cashflow_features,estimations_features,balance_features,hist_data_df_for_stock[stock])

                

                        ['price', 'evebitda', 'sma_50d', 'marketCap', 'EV', 'max_in_2W', 
                         'markRevRatio', 'peg', 'EVRevenues', 'fwdPriceTosale', 
                         'fwdPriceTosale_diff', 'var_sma100D', 'var_sma50D_100D',
                           'var_sma10D_100D', '1y_return', '1Y_6M_growth'] 
                        if prediction ==1 and  buy_probability>=0.9784:
                            if not GRID_SEARCH and i%DEBUG_FREQUENCY==0:
                                print(f"we will buy {stock} (prob = {buy_probability})")
                            stocks_in_range.append((stock, price_at_date,ev_ebitda,future_eps))
                    else:
                        stocks_in_range.append((stock, price_at_date,ev_ebitda,future_eps))
                    

        
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
                            'evebitda':stock_to_buy[2],
                            'future_eps':stock_to_buy[3],
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
                            'evebitda':stock_to_buy[2],
                            'future_eps':stock_to_buy[3],
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
        if nbr_total>0 and len(unique_stocks)>11 and len(unique_stocks)<30: #this is hardocded if we sample 250 stocks for the simulation
            score = (nbr_good-nbr_bad)/((nbr_total)*WAITING_IN_WEEKS)
            if score >-10:
                print(f"   ->score : {score}  mean : {mean}  uniquestocks = {len(unique_stocks)}")
        else:
            score=-99999
        return score
    
    
def simulation():
    data_list = []
    nbr_good=0
    nbr_bad=0
    print(f"simulation( SELL_LIMIT_PERCENTAGE_1 = {SELL_LIMIT_PERCENTAGE_1} , WAITING_IN_WEEKS = {WAITING_IN_WEEKS} , RATIO_UPPER_BOUND = {RATIO_UPPER_BOUND}")
    #delta_days = (max_random_date - min_random_date).days
    for i,random_date in enumerate(random_dates):
        random_date = random_date.date()
        gain_on_that_simul=0
        date_after_1W = random_date + pd.DateOffset(weeks=WAITING_IN_WEEKS)
        nbr_of_None_prices = 0
        nbr_of_None_pr_ratio = 0
        nbr_of_None_revenues = 0
        if not GRID_SEARCH:
            if i%10==0:
                print(f"Simulation {i+1}: {random_date}")

        stocks_in_peg_range = []
        for stock in stocks:
            if historical_data_for_stock[stock] is None or hist_data_df_for_stock[stock] is None or market_cap_dict[stock] is None or revenues_dict[stock] is None:
                if revenues_dict[stock] is None:
                    nbr_of_None_revenues+=1
                if not GRID_SEARCH:
                    if i%10==0:
                        print("one of the input data is None")
                continue
            price_at_date = get_stock_price_at_date(stock,random_date,historical_data_for_stock[stock])
            if price_at_date is None:
                nbr_of_None_prices+=1
                continue
            #peg_ratio = calculate_peg_ratio(stock,random_date,price_at_date,eps_calculation='quarter',data_key_metrics=keymetrics_for_stock[stock])
            pr_ratio = calculate_historical_price_to_revenue_ratio(stock,random_date,market_cap_dict[stock],revenues_dict[stock],do_api_call=False)
            if pr_ratio is None:
                nbr_of_None_pr_ratio+=1
            if pr_ratio is not None and pr_ratio>=0 and pr_ratio>=RATIO_LOWER_BOUND and pr_ratio<=RATIO_UPPER_BOUND:
                if USE_ML:
                    features_for_pred = ['revenues', 'price', 'pe_ratio', 'ebitdaMargin', 'ratio', 'sma_100d', 'sma_200d']
                    prediction, buy_probability = predict_buy(model,scaler,random_date,stock,features_for_pred,price_at_date,pr_ratio,sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict)
                    # features_for_pred = ['MarkRevRatio', 'sma_50w', 'sma_200d', 'sma_100d', 'price', 'sma_10d']
                    prediction2, buy_probability = predict_buy(model2,scaler,random_date,stock,features_for_pred,price_at_date,pr_ratio,sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict)
                    
                    if prediction ==1 and prediction is not None and prediction2==1 and prediction2 is not None:
                         stocks_in_peg_range.append((stock, price_at_date,pr_ratio))
                else:
                    stocks_in_peg_range.append((stock, price_at_date,pr_ratio))

        if not GRID_SEARCH:
                if i%10==0:
                    print(f"  at that date, fraction of stocks having None revenus : {nbr_of_None_revenues/len(stocks)}")
                    if (len(stocks)-nbr_of_None_revenues)!=0:
                        print(f"                fraction of valid stocks having None prices : {nbr_of_None_prices/(len(stocks)-nbr_of_None_revenues)}")
                        print(f"                fraction of valid stocks having None pr ratio: {nbr_of_None_pr_ratio/(len(stocks)-nbr_of_None_revenues)}")
        if len(stocks_in_peg_range)<=0:
            continue
        stocks_in_peg_range.sort(key=lambda x: x[1])
        #check if we can divide INTITIAL_CAPTIAL to buy stocks with equal weight
        end_of_loop=(len(stocks_in_peg_range)==0)
        last_index=0
        
        budget_per_stock = INTITIAL_CAPTIAL/len(stocks_in_peg_range)

        while(end_of_loop==False and last_index<len(stocks_in_peg_range)):
            if(stocks_in_peg_range[last_index][1]>budget_per_stock):
                end_of_loop = True
            else:
                last_index+=1
        stocks_to_buy = stocks_in_peg_range[:last_index]
        for stock_to_buy in stocks_to_buy:
            buy = True
            
            stock = stock_to_buy[0]
            initial_price = stock_to_buy[1]

            quantity = int(budget_per_stock/initial_price)
            gain_for_this_stock=0
            if buy:
                max_price_1W = get_max_price_in_range(stock,random_date,date_after_1W,hist_data_df_for_stock[stock])
                max_percentage_increase = (max_price_1W - initial_price) / initial_price * 100
                price_after_1W = get_stock_price_at_date(stock,date_after_1W,historical_data_for_stock[stock])
                if price_after_1W is None:
                    continue
                if not GRID_SEARCH:
                    if i%10==0:
                        print(f"we buy {quantity} of {stock} at {initial_price}$ its max price is {max_price_1W} and after {WAITING_IN_WEEKS} week (on {date_after_1W}) it is {price_after_1W}")
                if max_percentage_increase >= SELL_LIMIT_PERCENTAGE_1:
                    gain_for_this_stock= (SELL_LIMIT_PERCENTAGE_1/100)*initial_price*quantity - 2*TRANSACITON_FEE
                else:
                    gain_for_this_stock = (price_after_1W - initial_price)*quantity - 2*TRANSACITON_FEE
                if not GRID_SEARCH:
                    if i%10==0:
                        print(f"   -> {gain_for_this_stock} $")
                if gain_for_this_stock >= quantity*initial_price*A_GOOD_RETURN:
                    nbr_good +=1
                    if not GRID_SEARCH:
                        data_list.append({
                            'date': random_date,
                            'symbol': stock,
                            'price': initial_price,
                            'ratio':stock_to_buy[2],
                            'to_buy': 1
                        })
                elif gain_for_this_stock < quantity*initial_price*A_BAD_RETURN:
                    nbr_bad+=1
                    if not GRID_SEARCH:
                        data_list.append({
                            'date': random_date,
                            'symbol': stock,
                            'price': initial_price,
                            'ratio':stock_to_buy[2],
                            'to_buy': 0
                        })
                gain_on_that_simul+=gain_for_this_stock    

        
        if (gain_on_that_simul!=0):
            if not GRID_SEARCH:
                if i%10==0:
                    print(f"  ->total result : {gain_on_that_simul} ")
            total_gain.append(gain_on_that_simul)
        
    if not GRID_SEARCH:
        training_data = pd.DataFrame(data_list)
        training_data['date'] = pd.to_datetime(training_data['date'])

        print("nbr good : ",nbr_good)
        print("nbr_bad : ",nbr_bad)
        csv_filename = 'training_data.csv'
        if NBR_OF_SIMULATION>=100:
            training_data.to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")
        return total_gain
    else:
        #total_gain = np.array(total_gain)
        #mean = np.mean(total_gain)
        #return mean/SAFE_ZONE
        print("   ->score : ",(nbr_good-nbr_bad)/(WAITING_IN_WEEKS))
        return (nbr_good-nbr_bad)/(WAITING_IN_WEEKS)

if not GRID_SEARCH:
    total_gain = simulation()
    #print(total_gain)
    total_gain = np.array(total_gain)
    mean = np.mean(total_gain)
    print("mean : ",mean)
    print(total_gain)

def grid_search_simulation():
    # Define the ranges for each parameter
    sell_limit_1_range = [9,10,11,12,13,14] 
    upper_bound_range = [1.2]
    waiting_in_weeks_range = [16]
    # Generate all combinations of parameters
    parameter_combinations = list(itertools.product(sell_limit_1_range,upper_bound_range,waiting_in_weeks_range))

    best_result = float('-inf')
    best_params = None

    for params in parameter_combinations:
        sell_limit_1 , ratio_upper_bound, waiting_in_weeks= params
        

        global SELL_LIMIT_PERCENTAGE_1, RATIO_UPPER_BOUND, WAITING_IN_WEEKS
        SELL_LIMIT_PERCENTAGE_1 = sell_limit_1
        RATIO_UPPER_BOUND = ratio_upper_bound
        WAITING_IN_WEEKS = waiting_in_weeks

        result = simulation()

        # Update the best result if the current result is better
        if result > best_result:
            best_result = result
            best_params = params

    return best_params, best_result

# # Run the grid search
if GRID_SEARCH:
    best_parameters, best_score = grid_search_simulation()

    print(f"Best parameters: "
        f"SELL_LIMIT_PERCENTAGE_1={best_parameters[0]:.2f}, "
        f"RATIO_UPPER_BOUND = {best_parameters[1]:.2f}",
        f"WAITING_IN_WEEKS={best_parameters[2]:.2f}")
    print(f"Best score: {best_score}")

