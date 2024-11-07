the goal of this experiment is to find what fundamental metric predict the most a 5% return in 1month


i_researchAndDevelopmentExpenses:
[0.4835085175788329, 0.5307748008689356, 0.5311143270622286]
{'best_fit': 'quadratic', 'score': 1.0, 'equation': 'y = -0.02346337854840482x² + 0.07072966183850751x + 0.48350851757883284'}
    Segment 1: 0.00 to 24256000.00
    Segment 2: 24256000.00 to 102603000.00
    Segment 3: 102603000.00 to 306000000.00

dividend_payout_ratio:
[0.44811683320522677, 0.4396617986164489, 0.4458109146810146, 0.4869230769230769, 0.5465742879137798, 0.5899409802412112]
{'best_fit': 'quadratic', 'score': 0.9824323921968475, 'equation': 'y = 0.008448518112242773x² + -0.011643437266528785x + 0.4445018257342228'}
    Segment 1: -0.71 to -0.52
    Segment 2: -0.52 to -0.40
    Segment 3: -0.40 to -0.31
    Segment 4: -0.31 to -0.22
    Segment 5: -0.22 to -0.09
    Segment 6: -0.09 to 0.00

est_rev_growth:
[0.4982394366197183, 0.46045694200351495, 0.4591747146619842, 0.45606326889279436, 0.4687774846086192, 0.4868421052631579, 0.5166959578207382, 0.5783450704225352]
{'best_fit': 'quadratic', 'score': 0.9825524611881011, 'equation': 'y = 0.006255894177604361x² + -0.032628748490549773x + 0.49529684414548064'}
    Segment 1: -0.06 to -0.02
    Segment 2: -0.02 to -0.00
    Segment 3: -0.00 to 0.01
    Segment 4: 0.01 to 0.02
    Segment 5: 0.02 to 0.04
    Segment 6: 0.04 to 0.05
    Segment 7: 0.05 to 0.08
    Segment 8: 0.08 to 0.11

6M_return:
[0.4960352422907489, 0.48546255506607927, 0.4691358024691358, 0.46255506607929514, 0.44757709251101324, 0.4532627865961199, 0.4643171806167401, 0.4687224669603524]
{'best_fit': 'quadratic', 'score': 0.934221603017475, 'equation': 'y = 0.002293018692221537x² + -0.020331051330631716x + 0.49941437661701965'}
    Segment 1: -13.83 to -7.83
    Segment 2: -7.83 to -3.31
    Segment 3: -3.31 to 0.99
    Segment 4: 0.99 to 4.85
    Segment 5: 4.85 to 8.53
    Segment 6: 8.53 to 12.73
    Segment 7: 12.73 to 17.91
    Segment 8: 17.91 to 25.22

    
--------------5% in 1 month ------------------------
FOR NOW we have these conditions (pick one)
1)dividend_payout_ratio >-0.178 
=> 0.576 of 1

2)dividend_payout_ratio >-0.178 AND price <= 45.295 
=> 0.64 of 1 (on 1359 samples)

3)dividend_payout_ratio >-0.178 AND (current_est_eps + price + (current_est_eps/estimated growth)) <= 61.63 
=> 0.635 of 1 (1917 samples)

4)dividend_payout_ratio >-0.177 AND estimated_rev_growth>0.028
=> 0.6 of 1 (3365 samples)

5)dividend_payout_ratio + fcf_yield >-0.118 AND est_rev_growth/pegRatio >0.009
=>0.635 of 1 (2191 samples)

6)dividend_payout_ratio>-0.152 AND researDEvExpenses>239500 
=>0.634 (1852 samples)

7)dividend_payout_ratio>-0.148 AND netdebt/market_cap<=0.068
=>0.65 (1527 samples)

8)dividend_payout_ratio>-0.449 AND evEbitda_ratio<=34.454
=>0.6 (2042 samples)


9)1Yreturn<=-20.54 OR (dividend_payout_ratio>-0.449 AND evEbitda_ratio<=24.373)
=>0.62 (2306 samples)

10)future_estimated_eps - EVEbitdaRatio >-31
=>0.6 (2042 samples)