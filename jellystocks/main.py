import time
import pandas as pd
import portfolio
import jellyfish_search as js
# import matplotlib.pyplot as plt

b = portfolio.Portfolio(['AAPL', 'IBM', 'GOOGL'],
                        '2022-01-01',
                        '2023-01-01',
                        lower_bound=0,
                        upper_bound=1,
                        risk_free_rate=0.02)


p = js.JellyfishOptimizer(4, 'sharpe', early_termination=True)

b.optimize(p)


b.display_optimization_results()


