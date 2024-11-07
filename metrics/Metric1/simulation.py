import random
from datetime import datetime, timedelta
from .utils import *
import itertools
import pickle

"""
to run : py -m metrics.Metric1.simulation  (if it doesnt work add.py)
"""

INTITIAL_CAPTIAL = 2000
TRANSACITON_FEE = 2
RATIO_LOWER_BOUND = 0
RATIO_UPPER_BOUND = 0.8 #0.25
NBR_OF_SIMULATION = 960
SELL_LIMIT_PERCENTAGE_1 = 13 
WAITING_IN_WEEKS = 24
A_GOOD_RETURN = 0.0955
A_BAD_RETURN = 0.05 #0.007
GRID_SEARCH = False
USE_ML = True
SAVE_ONLY_BAD_PRED=True
stocks = load_stocks(200,'C:/Users/aziz8/Documents/FinancialMetricsLab/stock_list.csv')
historical_data_for_stock={}
hist_data_df_for_stock = {}
eps_date_dict={}
income_dict={}
historical_data_for_stock, hist_data_df_for_stock,eps_date_dict,income_dict = fetch_stock_data(stocks)

if USE_ML:
    sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict ,market_cap_dict = fetch_data_for_ml(stocks)
    model,scaler,model2,scaler2,model3,scaler3,model4,scaler4 = load_models()
    

print("sleep just before simulation")
time.sleep(30)

today = pd.Timestamp.today()
total_gain=[]
min_random_date = today - pd.DateOffset(months=70)
max_random_date = today - pd.DateOffset(months=7)
date_range = pd.date_range(start=min_random_date, end=max_random_date, freq='D')
date_range = date_range[(date_range.weekday < 5) & (date_range.year != 2020)]
max_simulation = min(NBR_OF_SIMULATION,len(date_range)-1)
random_dates = random.sample(list(date_range), max_simulation)


def simulation():
    data_list = []
    nbr_good=0
    nbr_bad=0
    print(f"simulation( SELL_LIMIT_PERCENTAGE_1 = {SELL_LIMIT_PERCENTAGE_1} , WAITING_IN_WEEKS = {WAITING_IN_WEEKS} , RATIO_UPPER_BOUND = {RATIO_UPPER_BOUND}")

    for i,random_date in enumerate(random_dates):
        random_date = random_date.date()
        gain_on_that_simul=0
        date_after_1W = random_date + pd.DateOffset(weeks=WAITING_IN_WEEKS)
        if not GRID_SEARCH:
            if i%10==0:
                print(f"Simulation {i+1}: {random_date}")

        stocks_in_peg_range = []
        for stock in stocks:
            if historical_data_for_stock[stock] is None or hist_data_df_for_stock[stock] is None :
                if not GRID_SEARCH:
                    if i%10==0:
                        print("one of the input data is None")
                continue
            price_at_date = get_stock_price_at_date(stock,random_date,historical_data_for_stock[stock])
            peg_ratio = calculate_quarter_peg_ratio(stock,random_date,price_at_date,prefetched_data=eps_date_dict[stock])
            
            if peg_ratio is not None and peg_ratio>=0 and peg_ratio>=RATIO_LOWER_BOUND and peg_ratio<=RATIO_UPPER_BOUND:
                if USE_ML:
                    features_for_pred =['MarkRevRatio', 'sma_200d', 'price', 'sma_50w']
                    prediction, buy_probability = predict_buy(model,scaler,random_date,stock,peg_ratio,features_for_pred,price_at_date,sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict,market_cap_dict)
                    prediction2, buy_probability = predict_buy(model2,scaler2,random_date,stock,peg_ratio,['MarkRevRatio', 'sma_50w', 'sma_200d', 'sma_100d', 'price', 'sma_10d'],price_at_date,sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict,market_cap_dict)
                    prediction3, buy_probability = predict_buy(model3,scaler3,random_date,stock,peg_ratio,['MarkRevRatio', 'sma_200d', 'price', 'sma_50w', 'sma_100d', 'ratio'],price_at_date,sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict,market_cap_dict)
                    prediction4, buy_probability = predict_buy(model4,scaler4,random_date,stock,peg_ratio,['MarkRevRatio', 'sma_200d', 'price', 'sma_50w', 'sma_100d', 'ratio', 'ebitdaMargin'],price_at_date,sma_10d_dict,sma_50w_dict,sma_100d_dict ,sma_200d_dict ,income_dict,market_cap_dict)
                    
                    if prediction ==1 and prediction2==1 and prediction3==1 and prediction4==1:
                         stocks_in_peg_range.append((stock, price_at_date,peg_ratio))
                else:
                    stocks_in_peg_range.append((stock, price_at_date,peg_ratio))

        stocks_in_peg_range.sort(key=lambda x: x[1])
        #check if we can divide INTITIAL_CAPTIAL to buy stocks with equal weight
        end_of_loop=(len(stocks_in_peg_range)==0)
        last_index=0
        if len(stocks_in_peg_range)<=0:
            continue
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
    total_gain = np.array(total_gain)
    mean = np.mean(total_gain)
    print("mean : ",mean)
    print(total_gain)

def grid_search_simulation():
    # Define the ranges for each parameter
    sell_limit_1_range = [10,11,12,13,14] 
    upper_bound_range = [0.8]
    waiting_in_weeks_range = [24]
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


