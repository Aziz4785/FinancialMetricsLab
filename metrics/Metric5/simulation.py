import random
from datetime import datetime, timedelta
from .utils import *
import itertools
import pickle
"""
to run : py -m metrics.Metric5.simulation  (if it doesnt work add.py)
to make the most of the grid search
run it several time with not a big number of stocks (250 is good) and pick the 
most consistent combination of parameters
"""

MIN_BUDGET_BY_STOCK = 400
TRANSACITON_FEE = 1.5
#DIVPAYOUT_LOWER_BOUND = -0.02
#DIVPAYOUT_UPPER_BOUND = 0.1
EV_EBITDA_UP = 30
EV_EBITDA_LOW = -30
FUTURE_EPS_UP = 0.01
#REVGROWTH_LOW_BOUND = 0.09
NBR_OF_SIMULATION = 1200
SELL_LIMIT_PERCENTAGE_1 = 10.9
#PE_UPPER_BOUND = 26.41
#PE_LOWER_BOUND = -20
#DIVIDENDS_PAID_LOW_BOUND = -2500000.00
WAITING_IN_WEEKS = 8
A_GOOD_RETURN = 0.099
A_BAD_RETURN = 0.05 
GRID_SEARCH = False
SAVE_ONLY_BAD_PRED=False
USE_ML = False
DEBUG_FREQUENCY=5
stocks = load_stocks(500,'C:/Users/aziz8/Documents/FinancialMetricsLab/stock_list.csv')
historical_data_for_stock, market_cap_dict, income_dict, hist_data_df_for_stock,balance_dict,cashflow_dict,estimations_dict,sector_dict = fetch_stock_data(stocks)


print("sleep just before simulation")
time.sleep(30)

today = pd.Timestamp.today()

min_random_date = today - pd.DateOffset(months=69)
max_random_date = today - pd.DateOffset(weeks=9)

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
    print(f"simulation( SELL_LIMIT_PERCENTAGE_1 = {SELL_LIMIT_PERCENTAGE_1} , EV_EBITDA_UP = {EV_EBITDA_UP} , EV_EBITDA_LOW = {EV_EBITDA_LOW},  FUTURE_EPS_UP = {FUTURE_EPS_UP}")
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
                if income_dict[stock] is None:
                    nbr_of_None_income+=1
                if not GRID_SEARCH:
                    if i%DEBUG_FREQUENCY==0:
                        print("one of the input data is None")
                continue
            price_at_date = extract_stock_price_at_date(random_date,historical_data_for_stock[stock])
            if price_at_date is None:
                nbr_of_None_prices+=1
                continue
            
            #dividend_payout_ratio = calculate_div_payout_ratio(random_date,cashflow_dict[stock],income_dict[stock])
            #pe_ratio = calculate_pe_ratio(random_date,price_at_date,income_dict[stock])
            ev_ebitda = calculate_evebitda(random_date,market_cap_dict[stock],balance_dict[stock],income_dict[stock])
            #revenue_est_growth = calculate_revenue_growth(random_date,estimations_dict[stock])
            #dividendsPaid = calculate_dividendPaid(random_date,cashflow_dict[stock])
            current_estimated_eps, future_eps = extract_current_and_future_estim_eps(random_date, estimations_dict[stock])
            #future_eps = calculate_FutureEps(random_date,estimations_dict[stock])
            #if dividend_payout_ratio is None:
                #nbr_of_None_evgp_ratio+=1
            if ev_ebitda is not None and ev_ebitda>=EV_EBITDA_LOW and ev_ebitda<=EV_EBITDA_UP:
                if future_eps is not None and future_eps<=FUTURE_EPS_UP :
                    stocks_in_range.append((stock, price_at_date,ev_ebitda,future_eps))
                    

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
        if not GRID_SEARCH:
            if i%DEBUG_FREQUENCY==0:
                print("stocks to buy : ",stocks_to_buy)
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
        training_data = pd.DataFrame(data_list)
        training_data['date'] = pd.to_datetime(training_data['date'])

        print("nbr good : ",nbr_good)
        print("nbr_bad : ",nbr_bad)
        csv_filename = 'training_data.csv'
        print(f"unique stocks  : {len(unique_stocks)}")
        if NBR_OF_SIMULATION>=100:
            training_data.to_csv(csv_filename, index=False)
            print(f"Data saved to {csv_filename}")
        return total_gain
    else:
        total_gain = np.array(total_gain)
        mean = np.mean(total_gain)
        if nbr_total>0 and len(unique_stocks)>11:
            score = (nbr_good-nbr_bad)/((nbr_total)*WAITING_IN_WEEKS)
            if score >-10:
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
    sell_limit_1_range = [10.9,11] 
    firstParam_upper_bound_range = [25,30]
    firstParam_lower_bound_range = [-40,-30]
    secondParam_lower_bound_range = [0.01,0.1,0.2]
    waiting_in_weeks_range = [8]
    # Generate all combinations of parameters
    parameter_combinations = list(itertools.product(sell_limit_1_range,firstParam_upper_bound_range,firstParam_lower_bound_range,secondParam_lower_bound_range,waiting_in_weeks_range))

    best_result = float('-inf')
    best_params = None

    for params in parameter_combinations:
        sell_limit_1 , firstPatam_up,firstPatam_low, secondParam_up, waiting_in_weeks= params
        

        global SELL_LIMIT_PERCENTAGE_1,EV_EBITDA_UP, EV_EBITDA_LOW,FUTURE_EPS_UP, WAITING_IN_WEEKS
        SELL_LIMIT_PERCENTAGE_1 = sell_limit_1
        EV_EBITDA_UP = firstPatam_up
        EV_EBITDA_LOW = firstPatam_low
        FUTURE_EPS_UP = secondParam_up

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
        f"EV_EBITDA_UP={best_parameters[1]:.2f}",
        f"EV_EBITDA_LOW={best_parameters[2]:.2f}",
        f"FUTURE_EPS_UP = {best_parameters[3]:.2f}",
        f"WAITING_IN_WEEKS={best_parameters[4]:.2f}")
    print(f"Best score: {best_score}")

