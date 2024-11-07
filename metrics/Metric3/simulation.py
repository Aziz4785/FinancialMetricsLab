import random
from datetime import datetime, timedelta
from .utils import *
import itertools
import pickle

"""
THIS DOESNT GIVE GOOD RESULTS :
CONCLUSION : EV/GP sucks ! (the explanation is that high EV/GP doesn't necessarly mean abad thing : it means the company has a good growth potential)

to run : py -m metrics.Metric3.simulation  (if it doesnt work add.py)
"""

#INTITIAL_CAPTIAL = 2000
MAX_BUDGET_BY_STOCK = 1000
TRANSACITON_FEE = 2
RATIO_LOWER_BOUND = 4
RATIO_UPPER_BOUND = 8
NBR_OF_SIMULATION = 600
SELL_LIMIT_PERCENTAGE_1 = 10.6
WAITING_IN_WEEKS = 14
A_GOOD_RETURN = 0.1
A_BAD_RETURN = 0.04 
GRID_SEARCH = False
SAVE_ONLY_BAD_PRED=True
USE_ML = True
stocks = load_stocks(250,'C:/Users/aziz8/Documents/FinancialMetricsLab/stock_list.csv')
market_cap_dict={}
income_dict={}
historical_data_for_stock={}
balance_shit_dict={}
historical_data_for_stock, market_cap_dict, income_dict, hist_data_df_for_stock,balance_shit_dict = fetch_stock_data(stocks)

if USE_ML:
    sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,sma_50d_dict,_,_,cashflow_dict = fetch_data_for_ml(stocks,income_mc=False)
    model,scaler,model2,scaler2,model3,scaler3,model4,scaler4 = load_models()
    

print("sleep just before simulation")
time.sleep(30)

today = pd.Timestamp.today()

min_random_date = today - pd.DateOffset(months=70)
max_random_date = today - pd.DateOffset(months=4)

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
            if i%10==0:
                print(f"Simulation {i+1}: {random_date}")

        stocks_in_peg_range = []
        for stock in stocks:
            if historical_data_for_stock[stock] is None or hist_data_df_for_stock[stock] is None or market_cap_dict[stock] is None or income_dict[stock] is None:
                if income_dict[stock] is None:
                    nbr_of_None_income+=1
                if not GRID_SEARCH:
                    if i%10==0:
                        print("one of the input data is None")
                continue
            price_at_date = get_stock_price_at_date(stock,random_date,historical_data_for_stock[stock])
            if price_at_date is None:
                nbr_of_None_prices+=1
                continue
            market_cap_at_date = get_historical_market_cap(random_date,stock,sorted_data=market_cap_dict[stock],do_api_call=False)

            evgp_ratio = calculate_evgp_ratio(stock, market_cap_at_date, random_date,balance_shit_dict[stock],income_dict[stock],do_api_call=False)
            if evgp_ratio is None:
                nbr_of_None_evgp_ratio+=1
            if evgp_ratio is not None and evgp_ratio>=0 and evgp_ratio>=RATIO_LOWER_BOUND and evgp_ratio<=RATIO_UPPER_BOUND:
                if USE_ML:
                    features1=['price', 'MarkRevRatio', 'sma_200d', 'ratio', 'fcf', 'sma_50w']
                    prediction, buy_probability = predict_buy(model,scaler,random_date,stock,evgp_ratio,features1,price_at_date,sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict,market_cap_dict,cashflow_dict)
                    features2=['MarkRevRatio', 'ratio', 'sma_200d', 'peRatio', 'sma_50w', 'sma_100d', 'price', 'other_expenses']
                    prediction2, buy_probability = predict_buy(model2,scaler2,random_date,stock,evgp_ratio,features2,price_at_date,sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict,market_cap_dict,cashflow_dict)
                    features3=['MarkRevRatio', 'ratio', 'sma_200d', 'peRatio', 'sma_50w', 'sma_100d']
                    prediction3, buy_probability = predict_buy(model3,scaler3,random_date,stock,evgp_ratio,features3,price_at_date,sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict,market_cap_dict,cashflow_dict)
                    features4 = ['peRatio', 'ratio', 'MarkRevRatio', 'price', 'sma_200d', 'sma_50w', 'fcf', 'sma_100d', 'ocfToNetinc', 'price_to_50w']
                    prediction4, buy_probability = predict_buy(model4,scaler4,random_date,stock,evgp_ratio,features4,price_at_date,sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict,market_cap_dict,cashflow_dict)
                    if not GRID_SEARCH:
                        if i%10==0:
                            print(f"{stock} has a good evgp, its predictions are {prediction} {prediction2} {prediction3}")
                    if prediction ==1 and prediction2==1 and prediction3==1 and prediction4==1:
                         stocks_in_peg_range.append((stock, price_at_date,evgp_ratio))
                else:
                    stocks_in_peg_range.append((stock, price_at_date,evgp_ratio))


        if not GRID_SEARCH:
                if i%10==0:
                    print(f"  at that date, fraction of stocks having None incomes : {nbr_of_None_income/len(stocks)}")
                    if (len(stocks)-nbr_of_None_income)!=0:
                        print(f"                fraction of valid stocks having None prices : {nbr_of_None_prices/(len(stocks)-nbr_of_None_income)}")
                        print(f"                fraction of valid stocks having None pr ratio: {nbr_of_None_evgp_ratio/(len(stocks)-nbr_of_None_income)}")
        if len(stocks_in_peg_range)<=0:
            continue
        stocks_in_peg_range.sort(key=lambda x: x[1])

        #budget_per_stock = INTITIAL_CAPTIAL/len(stocks_in_peg_range)
        budget_per_stock = MAX_BUDGET_BY_STOCK

        stocks_to_buy = stocks_in_peg_range
        for stock_to_buy in stocks_to_buy:
            buy = True
            
            stock = stock_to_buy[0]
            initial_price = stock_to_buy[1]

            quantity = int(budget_per_stock/initial_price)
            gain_for_this_stock=0
            if buy:
                max_price_1W = get_max_price_in_range(stock,random_date,date_after_1W,hist_data_df_for_stock[stock],do_api_call=False)
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
                    if not GRID_SEARCH and not SAVE_ONLY_BAD_PRED:
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
                nbr_total+=1
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
        total_gain = np.array(total_gain)
        mean = np.mean(total_gain)
        print(f"   ->score : {(nbr_good-nbr_bad)/((nbr_total)*WAITING_IN_WEEKS)}  mean : {mean}")
        return (nbr_good-nbr_bad)/((nbr_total)*WAITING_IN_WEEKS)

if not GRID_SEARCH:
    total_gain = simulation()
    #print(total_gain)
    total_gain = np.array(total_gain)
    mean = np.mean(total_gain)
    print("mean : ",mean)
    print(total_gain)

def grid_search_simulation():
    # Define the ranges for each parameter
    sell_limit_1_range = [10.5,11,11.5] 
    upper_bound_range = [8]
    waiting_in_weeks_range = [14]
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
    print("grid search")
    best_parameters, best_score = grid_search_simulation()

    print(f"Best parameters: "
        f"SELL_LIMIT_PERCENTAGE_1={best_parameters[0]:.2f}, "
        f"RATIO_UPPER_BOUND = {best_parameters[1]:.2f}",
        f"WAITING_IN_WEEKS={best_parameters[2]:.2f}")
    print(f"Best score: {best_score}")

#16 15weeks 12upperbound
#15 14weeeks 8 upperdon
#13 14weeks 4.5upperbond
#12 14weeks 4  score : -0.006057102644013271
#12 14weeks 3 score : 0.0055111