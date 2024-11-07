import unittest
from ..utils import *

"""
to run : py -m metrics.Metric2.tests.stock_price_test
"""
class TestGetStockPriceAtDate(unittest.TestCase):
        historical_data_for_stock = {}
        market_cap_dict={}
        revenues_dict = {}
        hist_data_df_for_stock ={}
        @classmethod
        def setUpClass(cls):
                # This will run once before all the test methods are executed
                stocks = ['AAPL','PFE','PLTR','WRB','RMD','LH','TTD']
                cls.historical_data_for_stock, cls.market_cap_dict, cls.revenues_dict, cls.hist_data_df_for_stock = fetch_stock_data(stocks)

        def test_valid_date(self):
                symbol = 'AAPL'
                date = '2024-08-15'
                result = get_stock_price_at_date(symbol, date,self.historical_data_for_stock[symbol])
                self.assertIsNotNone(result)

        def test_date_not_found(self):
                symbol = 'AAPL'
                date = '2026-08-16'
                result = get_stock_price_at_date(symbol, date)
                self.assertIsNone(result)

        def testPFE1(self):
                symbol = 'PFE'
                date = '2023-05-01'
                result = get_stock_price_at_date(symbol, date,self.historical_data_for_stock[symbol])
                self.assertIsNotNone(result)
                self.assertAlmostEqual(result, 39.21, delta=0.02)

        def testPFE2(self):
                symbol = 'PFE'
                date = '2023-05-02'
                result = get_stock_price_at_date(symbol, date,self.historical_data_for_stock[symbol])
                self.assertIsNotNone(result)
                self.assertAlmostEqual(result,  39.06, delta=0.02)

        def testPFE2b(self):
                symbol = 'PFE'
                date = '2023-05-02'
                result = get_stock_price_at_date(symbol, date,self.historical_data_for_stock[symbol])
                self.assertIsNotNone(result)
                self.assertAlmostEqual(round(result, 2), 39.06, places=2)

        def testPFE3(self):
                symbol = 'PFE'
                date = '2023-04-30'
                result = get_stock_price_at_date(symbol, date)
                self.assertIsNone(result)

        def testPFE3b(self):
                symbol = 'PFE'
                date = '2023-04-30'
                result = get_stock_price_at_date(symbol, date,self.historical_data_for_stock[symbol])
                self.assertIsNone(result)

        def testPFE4(self):
                symbol = 'PFE'
                date = '2023-04-29'
                result = get_stock_price_at_date(symbol, date)
                self.assertIsNone(result)

        def testPLTR1(self):
                symbol = 'PLTR'
                date = '2021-11-20'
                result = get_stock_price_at_date(symbol, date)
                self.assertIsNone(result)
        def testPLTR2(self):
                symbol = 'PLTR'
                date = '2021-11-14'
                result = get_stock_price_at_date(symbol, date)
                self.assertIsNone(result)

        def testPLTR3(self):
                symbol = 'PLTR'
                date = '2021-10-20'
                result = get_stock_price_at_date(symbol, date,self.historical_data_for_stock[symbol])
                self.assertAlmostEqual(round(result, 2), 24.22, places=2)

        def test5(self):
                symbol = 'RMD'
                date = '2019-08-12'
                print("for rmd : ")
                print(self.historical_data_for_stock[symbol])
                result = get_stock_price_at_date(symbol, date,self.historical_data_for_stock[symbol])
                self.assertIsNotNone(result)
                self.assertAlmostEqual(result,  131.61, delta=0.02)
        
        def test6(self):
                symbol = 'LH'
                date = '2019-08-12'
                result = get_stock_price_at_date(symbol, date,self.historical_data_for_stock[symbol])
                self.assertIsNotNone(result)
                self.assertAlmostEqual(result,  140.82, delta=0.02)
        def test7(self):
                symbol = 'TTD'
                date = '2019-08-12'
                result = get_stock_price_at_date(symbol, date,self.historical_data_for_stock[symbol])
                self.assertIsNotNone(result)
                self.assertAlmostEqual(result,  25.51, delta=0.02)
if __name__ == '__main__':
    unittest.main()