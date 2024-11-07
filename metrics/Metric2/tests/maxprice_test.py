import unittest
from ..utils import *

"""
to run : py -m metrics.Metric2.tests.maxprice_test
"""
class TestGetMaxPrice(unittest.TestCase):
        historical_data_for_stock = {}
        market_cap_dict={}
        revenues_dict = {}
        hist_data_df_for_stock ={}
        @classmethod
        def setUpClass(cls):
                # This will run once before all the test methods are executed
                stocks = ['AAPL','PFE','PLTR','WRB','RMD','LH','TTD']
                cls.historical_data_for_stock, cls.market_cap_dict, cls.revenues_dict, cls.hist_data_df_for_stock = fetch_stock_data(stocks)

        def test1(self):
                symbol = 'AAPL'
                date1 = '2024-08-15'
                date2 = '2024-10-15'
                result = get_max_price_in_range(symbol,date1,date2,self.hist_data_df_for_stock[symbol])
                self.assertIsNotNone(result) 
                self.assertAlmostEqual(result, 233.85, delta=0.02)
        def test2(self):
                symbol = 'PFE'
                date1 = '2023-02-10'
                date2 = '2024-10-15'
                result = get_max_price_in_range(symbol,date1,date2,self.hist_data_df_for_stock[symbol])
                self.assertIsNotNone(result) 
                self.assertAlmostEqual(result, 43.99, delta=0.02)
        def test3(self):
                symbol = 'PLTR'
                date1 = '2022-09-01'
                date2 = '2022-09-03'
                result = get_max_price_in_range(symbol,date1,date2,self.hist_data_df_for_stock[symbol])
                self.assertIsNotNone(result) 
                self.assertAlmostEqual(result, 7.4, delta=0.02)
        def test4(self):
                symbol = 'WRB'
                date1 = '2020-04-24'
                date2 = '2020-05-27'
                result = get_max_price_in_range(symbol,date1,date2,self.hist_data_df_for_stock[symbol])
                self.assertIsNotNone(result) 
                self.assertAlmostEqual(result, 26.02, delta=0.02)
        def test5(self):
                symbol = 'RMD'
                date1 = '2021-07-26'
                date2 = '2024-07-30'
                result = get_max_price_in_range(symbol,date1,date2,self.hist_data_df_for_stock[symbol])
                self.assertIsNotNone(result) 
                self.assertAlmostEqual(result, 297.36, delta=0.02)
        def test6(self):
                symbol = 'LH'
                date1 = '2023-11-15'
                date2 = '2023-12-20'
                result = get_max_price_in_range(symbol,date1,date2,self.hist_data_df_for_stock[symbol])
                self.assertIsNotNone(result) 
                self.assertAlmostEqual(result, 224.05, delta=0.02)

if __name__ == '__main__':
    unittest.main()
        