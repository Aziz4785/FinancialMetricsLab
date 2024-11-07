import unittest
from ..utils import *

"""
to run : py -m metrics.Metric1.tests.eps_test
https://www.alphaquery.com/subscribe
"""
class TestEPS(unittest.TestCase):
        historical_data_for_stock={}
        hist_data_df_for_stock = {}
        eps_date_dict={}
        income_dict={}
        @classmethod
        def setUpClass(cls):
                # This will run once before all the test methods are executed
                stocks = ['RMD','V','F','AAL','MSFT','KSS']
                cls.historical_data_for_stock, cls.hist_data_df_for_stock,cls.eps_date_dict,cls.income_dict = fetch_stock_data(stocks)
        def test1(self):
                symbol = 'RMD'
                date1 = '2023-10-26'
                current_quarter_eps,future_quarter_eps = get_current_and_future_estim_eps(symbol, date1,prefetched_sorted_data=self.eps_date_dict[symbol],income_state=self.income_dict[symbol])
                self.assertIsNotNone(current_quarter_eps) 
                self.assertIsNotNone(future_quarter_eps) 
                self.assertAlmostEqual(current_quarter_eps, 1.49, delta=0.02)
                self.assertAlmostEqual(future_quarter_eps, 1.776, delta=0.02)
        
        def test2(self):
                symbol = 'V'
                date1 = '2019-07-14'
                current_quarter_eps,future_quarter_eps = get_current_and_future_estim_eps(symbol, date1,prefetched_sorted_data=self.eps_date_dict[symbol],income_state=self.income_dict[symbol])
                self.assertIsNotNone(current_quarter_eps) 
                self.assertIsNotNone(future_quarter_eps) 
                self.assertAlmostEqual(current_quarter_eps, 1.37, delta=0.02)
                self.assertAlmostEqual(future_quarter_eps, 1.4277, delta=0.02)

        def test3(self):
                symbol = 'V'
                date1 = '2022-01-02'
                current_quarter_eps,future_quarter_eps = get_current_and_future_estim_eps(symbol, date1,prefetched_sorted_data=self.eps_date_dict[symbol],income_state=self.income_dict[symbol])
                self.assertIsNotNone(current_quarter_eps) 
                self.assertIsNotNone(future_quarter_eps) 
                self.assertAlmostEqual(current_quarter_eps, 1.81, delta=0.04)
                self.assertAlmostEqual(future_quarter_eps, 1.65137, delta=0.04)

        def test4(self):
                symbol = 'V'
                date1 = '2024-08-24'
                current_quarter_eps,future_quarter_eps = get_current_and_future_estim_eps(symbol, date1,prefetched_sorted_data=self.eps_date_dict[symbol],income_state=self.income_dict[symbol])
                self.assertIsNotNone(current_quarter_eps) 
                self.assertIsNotNone(future_quarter_eps) 
                self.assertAlmostEqual(current_quarter_eps,2.40, delta=0.05)
                self.assertAlmostEqual(future_quarter_eps, 2.58014, delta=0.04)

        def test5(self):
                symbol = 'F'
                date1 = '2022-01-10'
                current_quarter_eps,future_quarter_eps = get_current_and_future_estim_eps(symbol, date1,prefetched_sorted_data=self.eps_date_dict[symbol],income_state=self.income_dict[symbol])
                self.assertIsNotNone(current_quarter_eps) 
                self.assertIsNotNone(future_quarter_eps) 
                self.assertAlmostEqual(current_quarter_eps,0.26, delta=0.05)
                self.assertAlmostEqual(future_quarter_eps, 0.36979, delta=0.04)
        
        def test6(self):
                symbol = 'MSFT'
                date1 = '2022-01-10'
                current_quarter_eps,future_quarter_eps = get_current_and_future_estim_eps(symbol, date1,prefetched_sorted_data=self.eps_date_dict[symbol],income_state=self.income_dict[symbol])
                self.assertIsNotNone(current_quarter_eps) 
                self.assertIsNotNone(future_quarter_eps) 
                self.assertAlmostEqual(current_quarter_eps,2.48, delta=0.05)
                self.assertAlmostEqual(future_quarter_eps, 2.18879, delta=0.04)
        
        def test7(self):
                symbol = 'KSS'
                date1 = '2022-01-10'
                current_quarter_eps,future_quarter_eps = get_current_and_future_estim_eps(symbol, date1,prefetched_sorted_data=self.eps_date_dict[symbol],income_state=self.income_dict[symbol])
                self.assertIsNotNone(current_quarter_eps) 
                self.assertIsNotNone(future_quarter_eps) 
                self.assertAlmostEqual(current_quarter_eps,1.65, delta=0.05)
                self.assertAlmostEqual(future_quarter_eps, 2.10345, delta=0.04)
        
        def test8(self):
                symbol = 'MSFT'
                date1 = '2022-11-14'
                current_quarter_eps,future_quarter_eps = get_current_and_future_estim_eps(symbol, date1,prefetched_sorted_data=self.eps_date_dict[symbol],income_state=self.income_dict[symbol])
                self.assertIsNotNone(current_quarter_eps) 
                self.assertIsNotNone(future_quarter_eps) 
                self.assertAlmostEqual(current_quarter_eps,2.35, delta=0.05)
                self.assertAlmostEqual(future_quarter_eps, 2.29332, delta=0.04)
        
        def test9(self):
                symbol = 'AAL'
                date1 = '2023-10-27'
                current_quarter_eps,future_quarter_eps = get_current_and_future_estim_eps(symbol, date1,prefetched_sorted_data=self.eps_date_dict[symbol],income_state=self.income_dict[symbol])
                self.assertIsNotNone(current_quarter_eps) 
                self.assertIsNotNone(future_quarter_eps) 
                self.assertAlmostEqual(current_quarter_eps,0.38, delta=0.05)
                self.assertAlmostEqual(future_quarter_eps, 0.10687, delta=0.04)
if __name__ == '__main__':
    unittest.main()