est_rev_growth:
we divide the values of est_rev_growth into equal sized segments:
    Segment 1: -0.09 to -0.04
    Segment 2: -0.04 to -0.01
    Segment 3: -0.01 to 0.01
    Segment 4: 0.01 to 0.02
    Segment 5: 0.02 to 0.04
    Segment 6: 0.04 to 0.06
    Segment 7: 0.06 to 0.09
    Segment 8: 0.09 to 0.15
then for each segment we calculate the proportion of target==1 :
[np.float64(0.3913630229419703), np.float64(0.3784698713608666), np.float64(0.3813387423935091), np.float64(0.4129993229519296), np.float64(0.40040650406504064), np.float64(0.437542201215395), np.float64(0.47158322056833557), np.float64(0.5162381596752368)]
then we see if this follow a pattern
{'best_fit': 'quadratic', 'score': 0.9683379745284111, 'equation': 'y = 0.004045175025821949x² + -0.010510150648027712x + 0.3897375949627484'}

for data having est revenue >=0.09 :
-eps between -0.45 to -0.04 => score : 0.9539
-e_future_estimated_eps between -0.05 to 0.27 => score : 0.9351
-price to sales : 3.30 to 5.16 =>score : 0.9269
-pe ratio beween   -101.03 to -10.33 => score : 0.9302
-pegRatio between : -0.94 to -0.58 => score : 0.927
-dividen payout ratio : -0.13 to 0.00 =>score : 0.946771


to test :
df['EVEbitdaRatio'] <= 25.16 AND e_future_estimated_eps <=-0.08
df['EVEbitdaRatio'] <= 25.16 AND  dividend_payout_ratio>=0
df['EVEbitdaRatio'] <= 25.16 AND  div_by_rev_growth between 0.00 to 0.20