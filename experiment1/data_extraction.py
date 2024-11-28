import random
from datetime import datetime, timedelta
import itertools
import pickle
import pandas as pd 
from .utils import *

"""
to run : py -m experiment1.data_extraction  (if it doesnt work add.py)
"""

NBR_OF_SIMULATION = 500
WAITING_IN_WEEKS = 8
GOOD_RETURN = 0.09
BAD_RETURN = 0.5
if 'df_results' not in locals():
    df_results = pd.DataFrame()

stocks = load_stocks(510,'C:/Users/aziz8/Documents/FinancialMetricsLab/stock_list.csv')
print("stocks : ")
print(stocks)
historical_data_for_stock, market_cap_dict, income_dict, hist_data_df_for_stock,balance_shit_dict,cashflow_dict,estimations_dict,old_prices_dict = fetch_stock_data(stocks)

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
    date_after_1M = random_date + pd.DateOffset(weeks=WAITING_IN_WEEKS)
    random_stocks = random.sample(stocks, 48)

    for stock in random_stocks:
        price_at_date = extract_stock_price_at_date(random_date,historical_data_for_stock[stock])

        day1yago=random_date - pd.DateOffset(weeks=51)
        price_one_year_ago = extract_stock_price_at_date(day1yago,old_prices_dict[stock],not_None=True)
        if price_one_year_ago is None:
            price_one_year_ago = extract_stock_price_at_date(day1yago,historical_data_for_stock[stock],not_None=True)

        day6Mago=random_date - pd.DateOffset(weeks=24)
        price_6M_ago = extract_stock_price_at_date(day6Mago,old_prices_dict[stock],not_None=True)
        if price_6M_ago is None:
            price_6M_ago = extract_stock_price_at_date(day6Mago,historical_data_for_stock[stock],not_None=True)
        if price_6M_ago is None:
            continue
        
        if price_at_date is None or price_one_year_ago is None or price_at_date*price_one_year_ago ==0:
            continue
        max_price_1M = extract_max_price_in_range(random_date,date_after_1M,hist_data_df_for_stock[stock])
        if max_price_1M is None:
            continue
        percentage = (max_price_1M-price_at_date)/price_at_date
        to_buy=None
        if percentage>=GOOD_RETURN:
            to_buy=1
        elif percentage<=BAD_RETURN:
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
                'price_6M':price_6M_ago
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
            row_dict['est_eps_growth'] = (row_dict['e_future_estimated_eps']-row_dict['e_current_estimated_eps']) / row_dict['e_current_estimated_eps'] if row_dict['e_current_estimated_eps'] != 0 else None
            row_dict['peRatio'] = row_dict['price'] / row_dict['i_eps'] if row_dict['i_eps'] != 0 else None
            row_dict['pegRatio'] = row_dict['peRatio'] / (row_dict['est_eps_growth']*100) if (row_dict['est_eps_growth'] != 0 and row_dict['est_eps_growth'] is not None) else None
            row_dict['dividend_payout_ratio'] = row_dict['c_dividendsPaid'] / row_dict['i_netIncome'] if row_dict['i_netIncome'] != 0 else None
            row_dict['est_rev_growth'] = (row_dict['e_future_estim_rev']-row_dict['e_current_estim_rev']) / row_dict['e_current_estim_rev'] if row_dict['e_current_estim_rev'] != 0 else None
            row_dict['fcf'] = row_dict['c_operatingCashFlow']-row_dict['c_capitalExpenditure']
            row_dict['fcf_yield'] = row_dict['fcf'] / row_dict['market_cap'] if row_dict['market_cap'] != 0 else None
            row_dict['fcf_margin'] = row_dict['fcf'] / row_dict['i_revenue'] if row_dict['i_revenue'] != 0 else None
            row_dict['operating_cashflow'] = row_dict['c_netCashProvidedByOperatingActivities'] / row_dict['b_totalCurrentLiabilities'] if row_dict['b_totalCurrentLiabilities'] != 0 else None
            row_dict['cashflow_to_debt'] = row_dict['c_netCashProvidedByOperatingActivities'] / row_dict['b_totalDebt'] if row_dict['b_totalDebt'] != 0 else None

            row_dict['acid_test'] = (row_dict['b_cashAndShortTermInvestments']+row_dict['b_netReceivables']) / row_dict['b_totalCurrentLiabilities'] if row_dict['b_totalCurrentLiabilities'] != 0 else None
            row_dict['1Y_return'] = ((row_dict['price']-row_dict['price_1Y'])/row_dict['price_1Y'])*100
            row_dict['6M_return'] = ((row_dict['price']-row_dict['price_6M'])/row_dict['price_6M'])*100
            
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
            # Convert dictionary to DataFrame and append to existing DataFrame
            df_results = pd.concat([df_results, pd.DataFrame([row_dict])], ignore_index=True)

df_results.to_csv(F'experiment1/raw_data_{GOOD_RETURN*100}_in_{WAITING_IN_WEEKS}W.csv', index=False)