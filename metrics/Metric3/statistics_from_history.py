from .utils import *

"""
THIS DOESNT GIVE GOOD RESULTS :
CONCLUSION : EV/GP sucks !

to run : py -m metrics.Metric3.statistics_from_history  (if it doesnt work add .py)
"""

INCREASE_RANGE = '6months' #or '14weeks' or 1year
OPTION = 1 #put 1 or 3
stocks = load_stocks(300,'C:/Users/aziz8/Documents/FinancialMetricsLab/stock_list.csv')
print(stocks)
market_cap_dict={}
income_dict={}
historical_data_for_stock={}
balance_shit_dict={}
historical_data_for_stock, market_cap_dict, income_dict, hist_data_df_for_stock,balance_shit_dict = fetch_stock_data(stocks)


print("length of historical_data_for_stock = ",len(historical_data_for_stock))
none_count = sum(1 for value in historical_data_for_stock.values() if value is None)
print("number of None in the values of historical_data_for_stock = ", none_count)

data_to_analyze = []



def get_statistics_from_history():
    print("get_statistics_from_history()")
    today = datetime.now().date()
    if INCREASE_RANGE == '1year':
        one_year_ago = today - timedelta(days=370)
    elif INCREASE_RANGE == '3months':
        one_year_ago = today - timedelta(days=100)
    elif INCREASE_RANGE == '4months':
        one_year_ago = today - timedelta(days=130)
    elif INCREASE_RANGE == '5months':
        one_year_ago = today - timedelta(days=160)
    elif INCREASE_RANGE == '6months':
        one_year_ago = today - timedelta(days=190)

    date_range = pd.date_range(start='2019-10-10', end=one_year_ago)

    filtered_dates = []
    for date_ in date_range:
        if date_.year != 2020 and date_.weekday()<5:
            filtered_dates.append(date_.date())

    print("length of filtered dates: ",len(filtered_dates))
    print()
    counter_date=0

    # Print the result
    for date in filtered_dates:
        nbr_none_prices = 0
        nbr_none_evgp=0
        nbr_non_because_missing_date=0
        if counter_date%80 == 0:
            print(date)
        counter_date+=1
        for stock in stocks:
            historical_prices = historical_data_for_stock[stock]
            price_at_date = get_stock_price_at_date(stock,date,historical_prices)
            if price_at_date is None:
                nbr_none_prices+=1
                if historical_prices is not None and date not in historical_prices:
                    nbr_non_because_missing_date+=1
                continue
                
            market_cap_at_date = get_historical_market_cap(date,stock,sorted_data=market_cap_dict[stock],do_api_call=False)

            evgp_ratio = calculate_evgp_ratio(stock, market_cap_at_date, date,balance_shit_dict[stock],income_dict[stock],do_api_call=False)
            if evgp_ratio is None:
                nbr_none_evgp+=1
            if evgp_ratio is None or evgp_ratio<0 or evgp_ratio>=400:
                continue
            if INCREASE_RANGE == '1year':
                date_after_1Y = date + pd.DateOffset(years=1)
            elif INCREASE_RANGE == '3months':
                date_after_1Y = date + pd.DateOffset(months=3)
            elif INCREASE_RANGE == '14weeks':
                date_after_1Y = date + pd.DateOffset(weeks=14)
            elif INCREASE_RANGE =='6months':
                date_after_1Y = date + pd.DateOffset(months=6)
            elif INCREASE_RANGE =='4months':
                date_after_1Y = date + pd.DateOffset(months=4)
            elif INCREASE_RANGE == '5months':
                date_after_1Y = date + timedelta(days=160)

            max_price_1Y = get_max_price_in_range(stock,date,date_after_1Y,hist_data_df_for_stock[stock])
            
            if max_price_1Y is None or max_price_1Y==0:
                continue
            max_percentage_increase = (max_price_1Y-price_at_date) / price_at_date
            price_after_1Y = get_stock_price_at_date(stock,date_after_1Y,historical_prices,not_None=True)
            if price_after_1Y is None:
                #print("price_after_1Y is None")
                continue
            percentage_increase_1Y = (price_after_1Y-price_at_date) / price_at_date
            #print(f"stock : {stock} , price at that date = {price_at_date} peg_ratio = {peg_ratio} , max_price_1Y = {max_price_1Y} -> {max_percentage_increase*100}%, price_after_1Y = {price_after_1Y} ")
            data_to_analyze.append({
                        'date': date,
                        'stock': stock,
                        'evgp_ratio': evgp_ratio,
                        'price_at_date': price_at_date,
                        'max_percentage_increase': max_percentage_increase,
                        'percentage_increase_1Y': percentage_increase_1Y
                    })
        if counter_date%80 == 0:
            if nbr_none_prices/len(stocks) >=0.3:
                print(f"at date {date} percentage of None prices {nbr_none_prices*100/len(stocks)} %  AND PERCENTAGE OF nONE BECAUSE IT HAS A MISSING DATE : {nbr_non_because_missing_date*100/len(stocks)}") 
            if nbr_none_evgp/len(stocks) >=0.5:
                print(f"at date {date} more than 50% pf evgp are none") 
    print()
    return data_to_analyze


def analyze_and_visualize_peg_ratio_vs_max_increase(df,column='max_percentage_increase'):
    # Define PEG ratio ranges
    peg_ranges = [(0, 4),(4,8), (8, 12), (12, 16), (16, float('inf'))]

    # Function to assign PEG ratio to a range
    def assign_peg_range(peg):
        for i, (low, high) in enumerate(peg_ranges):
            if low <= peg < high:
                return f'{low}-{high}' if high != float('inf') else f'{low}+'
        return 'Unknown'

    # Add PEG range column to the DataFrame
    df['evgp_range'] = df['evgp_ratio'].apply(assign_peg_range)



    # Group by PEG range and calculate average max percentage increase
    grouped_mean = df.groupby('evgp_range')[column].mean().sort_index()
    grouped_median = df.groupby('evgp_range')[column].median().sort_index()
    grouped_min = df.groupby('evgp_range')[column].min().sort_index()

    # Plotting
    plt.figure(figsize=(14, 7))
    x = range(len(grouped_mean))
    width = 0.35

    plt.bar([i - width/2 for i in x], grouped_mean, width, label='Mean', color='skyblue')
    plt.bar([i + width/2 for i in x], grouped_median, width, label='Median', color='lightgreen')

    plt.title(f'Average and Median {column}e by EVGP Ratio Range')
    plt.xlabel('EVGP Ratio Range')
    plt.ylabel(column)
    plt.xticks(x, grouped_mean.index, rotation=45)
    plt.legend()
    plt.tight_layout()

    plt.show()

    # Print the data
    print("Mean values:")
    print(grouped_mean)
    print("\nMedian values:")
    print(grouped_median)
    print("\min values:")
    print(grouped_min)
    return grouped_mean, grouped_median,grouped_min

if OPTION ==1:
    print("option 1 : ")
    data_to_analyze = get_statistics_from_history()
    print("length of data (=nbr  of not null rows): ",len(data_to_analyze))
    df_data_to_analyze = pd.DataFrame(data_to_analyze)
    max_increase_results = analyze_and_visualize_peg_ratio_vs_max_increase(df_data_to_analyze)