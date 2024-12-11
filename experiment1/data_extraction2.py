import random
from datetime import datetime, timedelta
import itertools
import pickle
import pandas as pd 
from .utils import *

"""
THE DIFFERENCE WITH THE OTHER FILE IS THAT THIS ONE HAS A STOP LOSS
to run : py -m experiment1.data_extraction2  (if it doesnt work add.py)
"""

NBR_OF_SIMULATION = 500 #500
WAITING_IN_WEEKS = 6

BUDGET_BY_STOCK = 400
TAKE_PROFIT_DOLLARS = 36
STOP_LOSS_DOLLARS = -20
if 'df_results' not in locals():
    df_results = pd.DataFrame()

stocks = load_stocks(525,'stock_list.csv')
print("stocks : ")
print(stocks)
historical_data_for_stock,high_df,low_df, market_cap_dict, income_dict, hist_data_df_for_stock,balance_shit_dict,cashflow_dict,estimations_dict = fetch_stock_data(stocks)

today = pd.Timestamp.today()
min_random_date = today - pd.DateOffset(months=62)
max_random_date = today - pd.DateOffset(weeks=WAITING_IN_WEEKS+1)
date_range = pd.date_range(start=min_random_date, end=max_random_date, freq='D')
date_range = date_range[(date_range.weekday < 5) & (date_range.year != 2020)]
max_simulation = min(NBR_OF_SIMULATION,len(date_range)-1)
random_dates = random.sample(list(date_range), max_simulation)
print("length of random_date : ",len(random_dates))
for i,random_date in enumerate(random_dates):
    if i%100 ==0:
        print(i)
    
    random_date = random_date.date()
    gain_on_that_simul=0
    date_after_period = random_date + pd.DateOffset(weeks=WAITING_IN_WEEKS)
    random_stocks = random.sample(stocks, 49)
    #print(f"for the interval : {random_date} - {date_after_period}")
    for stock in random_stocks:
        #print("we pick this stock : ",stock)
        price_at_date = extract_stock_price_at_date(random_date,historical_data_for_stock[stock])

        day1yago=random_date - pd.DateOffset(weeks=51)
        price_one_year_ago = extract_stock_price_at_date(day1yago,historical_data_for_stock[stock],not_None=True)

        day6Mago=random_date - pd.DateOffset(weeks=24)
        price_6M_ago = extract_stock_price_at_date(day6Mago,historical_data_for_stock[stock],not_None=True)

        price_3M_ago = extract_stock_price_at_date(random_date - pd.DateOffset(weeks=12),historical_data_for_stock[stock],not_None=True)
        if price_6M_ago is None or price_3M_ago is None:
            continue
        
        if price_at_date is None or price_one_year_ago is None or price_at_date*price_one_year_ago ==0:
            continue
        quantity = int(BUDGET_BY_STOCK/price_at_date)
        if quantity ==0:
            continue
        actual_spent_budget = quantity*price_at_date
        target_price = TAKE_PROFIT_DOLLARS/quantity + price_at_date -0.01
        stop_loss_price = price_at_date + STOP_LOSS_DOLLARS/quantity + 0.01
        #max_price_1M,max_date = extract_max_price_in_range(random_date,date_after_period,high_df[stock],return_date=True)
        #min_price_1M,min_date = extract_min_price_in_range(random_date,date_after_period,low_df[stock],return_date=True)
        _,take_profit_date = find_first_price_threshold(random_date, date_after_period, high_df[stock], target_price,'profit', return_date=True)
        _,stop_loss_date = find_first_price_threshold(random_date, date_after_period, low_df[stock], stop_loss_price,'loss', return_date=True)
        
        #print("price at the beginning : ",price_at_date)
        #print("max price is : ",max_price_1M)
        #print("min price is : ",min_price_1M)

        #max_gain = (max_price_1M-price_at_date)*quantity
        #min_gain = (min_price_1M-price_at_date)*quantity
        #print(f"max price is : {max_price_1M} which represent an increase of {max_percentage*100} %")
        #print(f"min price is : {min_price_1M} which represent an decrease of {min_percentage*100} %")
        #print("so the loss is : ",loss)
        #print("and the gain is : ",gain)
        to_buy=None
        if stop_loss_date is not None and take_profit_date is not None and take_profit_date<stop_loss_date:
            to_buy=1
        elif stop_loss_date is not None and take_profit_date is not None and take_profit_date>stop_loss_date:
            to_buy=0
        elif stop_loss_date is None and take_profit_date is not None :
            to_buy=1
        elif stop_loss_date is not None and take_profit_date is None :
            to_buy=0

        income_features = extract_income_features(random_date,income_dict[stock])
        balance_features = extract_balance_features(random_date,balance_shit_dict[stock])
        cashflow_features = extract_cashflow_features(random_date,cashflow_dict[stock])
        estimation_features = extract_estimation_features(random_date,estimations_dict[stock])
        market_cap = extract_market_cap(random_date,market_cap_dict[stock])
        if to_buy is not None and income_features is not None and cashflow_features is not None and market_cap is not None and balance_features is not None and estimation_features is not None:
            row_dict = {
                'date': random_date,
                'stock': stock,
                'target': to_buy,
                'market_cap': market_cap,
                'price':price_at_date,
                'price_1Y':price_one_year_ago,
                'price_6M':price_6M_ago,
                'price_3M': price_3M_ago
            }
            # Add income features with prefix
            row_dict.update({f'i_{k}': v for k, v in income_features.items()})
            
            # Add balance features with prefix
            row_dict.update({f'b_{k}': v for k, v in balance_features.items()})
            
            # Add cashflow features with prefix
            row_dict.update({f'c_{k}': v for k, v in cashflow_features.items()})
            
            row_dict.update({f'e_{k}': v for k, v in estimation_features.items()})

            row_dict['price_to_sales'] = row_dict['market_cap'] / row_dict['i_revenue'] if row_dict['i_revenue'] != 0 else None
            row_dict['fwd_price_to_sales_diff'] = calculate_fxd_price_to_sales_diff(
                row_dict['market_cap'],
                row_dict['e_future_estim_rev'],
                row_dict['price_to_sales']
            )
            row_dict['eps_diff'] = safe_multiply(safe_divide(safe_subtract(row_dict['e_current_estimated_eps'],row_dict['i_eps']),row_dict['i_eps']),100)
            row_dict['est_eps_growth'] = safe_divide((row_dict['e_future_estimated_eps']-row_dict['e_current_estimated_eps']), row_dict['e_current_estimated_eps'])
            row_dict['peRatio'] = safe_divide(row_dict['price'],row_dict['i_eps'])
            row_dict['pegRatio'] = safe_divide(row_dict['peRatio'] , safe_multiply(row_dict['est_eps_growth'],100)) 
            row_dict['dividend_payout_ratio'] = safe_divide(row_dict['c_dividendsPaid'],row_dict['i_netIncome']) 
            row_dict['est_rev_growth'] = (row_dict['e_future_estim_rev']-row_dict['e_current_estim_rev']) / row_dict['e_current_estim_rev'] if row_dict['e_current_estim_rev'] != 0 else None
            row_dict['fcf'] = row_dict['c_operatingCashFlow']-row_dict['c_capitalExpenditure']
            row_dict['fcf_yield'] = safe_divide(row_dict['fcf'], row_dict['market_cap']) 
            row_dict['fcf_margin'] = safe_divide(row_dict['fcf'], row_dict['i_revenue'])
            row_dict['operating_cashflow'] = row_dict['c_netCashProvidedByOperatingActivities'] / row_dict['b_totalCurrentLiabilities'] if row_dict['b_totalCurrentLiabilities'] != 0 else None
            row_dict['cashflow_to_debt'] = row_dict['c_netCashProvidedByOperatingActivities'] / row_dict['b_totalDebt'] if row_dict['b_totalDebt'] != 0 else None

            row_dict['price_to_nbrshares']= safe_divide(row_dict['price'],safe_divide(row_dict['i_netIncome'],row_dict['i_eps']))
            row_dict['acid_test'] = (row_dict['b_cashAndShortTermInvestments']+row_dict['b_netReceivables']) / row_dict['b_totalCurrentLiabilities'] if row_dict['b_totalCurrentLiabilities'] != 0 else None
            row_dict['1Y_return'] = ((row_dict['price']-row_dict['price_1Y'])/row_dict['price_1Y'])*100
            row_dict['6M_return'] = ((row_dict['price']-row_dict['price_6M'])/row_dict['price_6M'])*100
            row_dict['3M_return'] = (safe_subtract(row_dict['price'],row_dict['price_3M'])/row_dict['price_3M'])*100
            
            row_dict['div_by_rev_growth'] = safe_division_ratio(
                row_dict['dividend_payout_ratio'],
                row_dict['est_rev_growth']
            )
            row_dict["GPRatio"] = calculate_growthpotential_ratio(
                row_dict['i_researchAndDevelopmentExpenses'],
                row_dict['c_freeCashFlow'],
                row_dict['i_revenue']
            )
            #row_dict['PriceToRD'] = row_dict['market_cap'] / row_dict['i_researchAndDevelopmentExpenses'] if row_dict['i_researchAndDevelopmentExpenses'] != 0 else None
            row_dict['nedDebtToMarketCap'] = row_dict['b_netDebt'] / row_dict['market_cap'] if row_dict['market_cap'] != 0 else None
            

            enterprise_value = row_dict['market_cap'] + row_dict["b_totalDebt"] - row_dict["b_cashAndCashEquivalents"]
            gross_profit = row_dict['i_grossProfit']
            row_dict['EVGPratio'] = enterprise_value/gross_profit if gross_profit!=0 else None
            row_dict['EVEbitdaRatio'] = enterprise_value/row_dict['i_ebitda'] if row_dict['i_ebitda']!=0 else None
            row_dict['EVRevenue'] = enterprise_value/row_dict['i_revenue'] if row_dict['i_revenue']!=0 else None
            row_dict['combined_valuation_score'] = safe_divide(safe_add(safe_add(row_dict['EVEbitdaRatio'],row_dict['peRatio']), safe_add(row_dict['pegRatio'],row_dict['price_to_sales'])),4)
            # Convert dictionary to DataFrame and append to existing DataFrame
            df_results = pd.concat([df_results, pd.DataFrame([row_dict])], ignore_index=True)

df_results.to_csv(F'experiment1/raw_data_{TAKE_PROFIT_DOLLARS}D_in_{WAITING_IN_WEEKS}W.csv', index=False)