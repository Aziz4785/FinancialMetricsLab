import random
from datetime import datetime, timedelta
from .utils import *
import itertools
import pickle
"""
to run : py -m metrics.Metric4.simulation  (if it doesnt work add.py)
to make the most of the grid search
run it several time with not a big number of stocks (250 is good) and pick the 
most consistent combination of parameters
"""

MIN_BUDGET_BY_STOCK = 1000
TRANSACITON_FEE = 2
DIVPAYOUT_LOWER_BOUND = -0.02
DIVPAYOUT_UPPER_BOUND = 0.1
REVGROWTH_LOW_BOUND = 0.09
NBR_OF_SIMULATION = 600
SELL_LIMIT_PERCENTAGE_1 = 5.7
WAITING_IN_WEEKS = 4
A_GOOD_RETURN = 0.05
A_BAD_RETURN = 0.025 
GRID_SEARCH = False
SAVE_ONLY_BAD_PRED=False
USE_ML = False
stocks = load_stocks(250,'C:/Users/aziz8/Documents/FinancialMetricsLab/stock_list.csv')
historical_data_for_stock, _, income_dict, hist_data_df_for_stock,_,cashflow_dict,estimations_dict = fetch_stock_data(stocks)


print("sleep just before simulation")
time.sleep(30)

today = pd.Timestamp.today()

min_random_date = today - pd.DateOffset(months=70)
max_random_date = today - pd.DateOffset(weeks=5)

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
    print(f"simulation( SELL_LIMIT_PERCENTAGE_1 = {SELL_LIMIT_PERCENTAGE_1} , DIVPAYOUT_UPPER_BOUND = {DIVPAYOUT_UPPER_BOUND} , DIVPAYOUT_LOWER_BOUND = {DIVPAYOUT_LOWER_BOUND},  REVGROWTH_LOW_BOUND = {REVGROWTH_LOW_BOUND}")
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

        stocks_in_range = []
        for stock in stocks:
            if historical_data_for_stock[stock] is None or hist_data_df_for_stock[stock] is None or income_dict[stock] is None:
                if income_dict[stock] is None:
                    nbr_of_None_income+=1
                if not GRID_SEARCH:
                    if i%10==0:
                        print("one of the input data is None")
                continue
            price_at_date = extract_stock_price_at_date(random_date,historical_data_for_stock[stock])
            if price_at_date is None:
                nbr_of_None_prices+=1
                continue
            
            dividend_payout_ratio = calculate_div_payout_ratio(random_date,cashflow_dict[stock],income_dict[stock])
            revenue_est_growth = calculate_revenue_growth(random_date,estimations_dict[stock])
            
            if dividend_payout_ratio is None:
                nbr_of_None_evgp_ratio+=1
            if dividend_payout_ratio is not None and dividend_payout_ratio>=DIVPAYOUT_LOWER_BOUND and dividend_payout_ratio<=DIVPAYOUT_UPPER_BOUND:
                if revenue_est_growth is not None and revenue_est_growth>=REVGROWTH_LOW_BOUND :
                    stocks_in_range.append((stock, price_at_date,dividend_payout_ratio,revenue_est_growth))
                    

        if not GRID_SEARCH:
                if i%10==0:
                    print(f"  at that date, fraction of stocks having None incomes : {nbr_of_None_income/len(stocks)}")
                    if (len(stocks)-nbr_of_None_income)!=0:
                        print(f"                fraction of valid stocks having None prices : {nbr_of_None_prices/(len(stocks)-nbr_of_None_income)}")
                        print(f"                fraction of valid stocks having None pr ratio: {nbr_of_None_evgp_ratio/(len(stocks)-nbr_of_None_income)}")
        if len(stocks_in_range)<=0:
            continue
        #stocks_in_range.sort(key=lambda x: x[1])

        #budget_per_stock = INTITIAL_CAPTIAL/len(stocks_in_range)
        budget_per_stock = MIN_BUDGET_BY_STOCK

        stocks_to_buy = stocks_in_range
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
                    unique_stocks.add(stock)
                    if not GRID_SEARCH and not SAVE_ONLY_BAD_PRED:
                        data_list.append({
                            'date': random_date,
                            'symbol': stock,
                            'price': initial_price,
                            'divpayout_ratio':stock_to_buy[2],
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
                            'divpayout_ratio':stock_to_buy[2],
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
        if nbr_total>0 and len(unique_stocks)>50:
            score = (nbr_good-nbr_bad)/((nbr_total)*WAITING_IN_WEEKS)
            if score >0:
                print(f"   ->score : {score}  mean : {mean}  uniquestocks = {len(unique_stocks)}")
        else:
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
    sell_limit_1_range = [5.7,5.8] # 5.8 5.8 5.7 5.7 5.8 5.8 5.7
    divpayout_upper_bound_range = [0.1,0.15] # 0.15 0.15 0.1 0.12 0.1 0.1 0.15
    divpayout_lower_bound_range = [-0.06,-0.05,-0.04,-0.03,-0.02]# -0.06 -0.06 -0.05 -0.04 -0.03 -0.06 -0.03
    rev_growth_lower_bound_range = [0.09,0.1,0.12]# 0.1 0.1 0.09 0.09 0.09 0.1 0.09
    waiting_in_weeks_range = [4]
    # Generate all combinations of parameters
    parameter_combinations = list(itertools.product(sell_limit_1_range,divpayout_upper_bound_range,divpayout_lower_bound_range,rev_growth_lower_bound_range,waiting_in_weeks_range))

    best_result = float('-inf')
    best_params = None

    for params in parameter_combinations:
        sell_limit_1 , divratio_upper_bound,divratio_lower_bound, revlow_bound, waiting_in_weeks= params
        

        global SELL_LIMIT_PERCENTAGE_1,DIVPAYOUT_UPPER_BOUND, DIVPAYOUT_LOWER_BOUND,REVGROWTH_LOW_BOUND, WAITING_IN_WEEKS
        SELL_LIMIT_PERCENTAGE_1 = sell_limit_1
        DIVPAYOUT_UPPER_BOUND = divratio_upper_bound
        DIVPAYOUT_LOWER_BOUND = divratio_lower_bound
        REVGROWTH_LOW_BOUND = revlow_bound

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
        f"DIVPAYOUT_UPPER_BOUND={best_parameters[1]:.2f}",
        f"DIVPAYOUT_LOWER_BOUND={best_parameters[2]:.2f}",
        f"REVGROWTH_LOW_BOUND = {best_parameters[3]:.2f}",
        f"WAITING_IN_WEEKS={best_parameters[4]:.2f}")
    print(f"Best score: {best_score}")

