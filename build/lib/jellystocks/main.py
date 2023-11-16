import time
import jellyfish_search as js
import portfolio

# import matplotlib.pyplot as plt

# b = portfolio.Portfolio(['AAPL', 'IBM', 'GOOGL'],
#                         '2022-01-01',
#                         '2023-01-01',
#                         lower_bound=0,
#                         upper_bound=1,
#                         risk_free_rate=0.02)

b = portfolio.Portfolio(
    ['AAPL', 'MSFT', 'INTC', 'IBM', 'CSCO', 'AMZN', 'DIS', 'MCD', 'NKE', 'HD', 'WMT', 'PG', 'KO', 'PEP'
        , 'WBA', 'CL', 'MO', 'JNJ', 'PFE', 'MRK', 'ABT', 'AMGN', 'UNH', 'GILD', 'JPM', 'BAC', 'WFC', 'C'
        , 'AXP', 'GE', 'BA', 'MMM', 'CAT', 'DE', 'RTX', 'XOM', 'CVX', 'SLB', 'COP', 'HAL', 'BP', 'DUK'
        , 'SO', 'D', 'NEE', 'EXC', 'T', 'VZ', 'DD', 'NEM'],
    '2013-01-01',
    '2023-01-01',
    0.0,
    1.0
)
# df = pd.DataFrame(b.data)
# df.to_csv('data.csv')

# print(f'risk free right when creating the object: {b.risk_free_rate} ')
# print(f'sharpe attribute BEFORE opt: {b.sharpe_ratio} ')
# print(f'expected returns attribute BEFORE opt: {b.annualized_expected_returns} ')
# print(f'expected returns attribute BEFORE opt: {b.portfolio_standard_deviation} ')
# print(f'log returns attribute BEFORE opt: {b._log_returns} ')
# print(f'cov mtx attribute BEFORE opt: {b._annual_cov_matrix} ')
# print(f'tickers BEFORE opt: {b.tickers} ')

# print(b.data.head(15))
# print(b._log_returns)
# print(b._annual_cov_matrix)
p = js.JellyfishOptimizer(20, 'sharpe', early_termination=True)

results = []

start_time = time.time()
for iteration in range(1):
    best_solution, best_cost, iterations, evaluations, elapsed_time, cost_over_time, _ = b.optimize(p, num_runs=10)
    results.append(best_cost)

elapsed_time = time.time() - start_time

b.display_optimization_results()

# best = min(results)
# print(f'best sharpe ratio: {best}')
# print(f'elapsed time: {elapsed_time}')
# print(f'sharpe attribute after opt: {b.sharpe_ratio} ')
# print(f'risk free attribute after opt: {b.risk_free_rate} ')
# for item in [best_solution, best_cost, iterations, evaluations, elapsed_time, cost_over_time]:
#     print(f' {item}')
# b = portfolio.Portfolio(
#     ['AAPL', 'MSFT', 'INTC', 'IBM', 'CSCO', 'AMZN', 'DIS', 'MCD', 'NKE', 'HD', 'WMT', 'PG', 'KO', 'PEP'
#         , 'WBA', 'CL', 'MO', 'JNJ', 'PFE', 'MRK', 'ABT', 'AMGN', 'UNH', 'GILD', 'JPM', 'BAC', 'WFC', 'C'
#         , 'AXP', 'GE', 'BA', 'MMM', 'CAT', 'DE', 'RTX', 'XOM', 'CVX', 'SLB', 'COP', 'HAL', 'BP', 'DUK'
#         , 'SO', 'D', 'NEE', 'EXC', 'T', 'VZ', 'DD', 'NEM'],
#     '2013-01-01',
#     '2023-01-01',
#     0.0,
#     1.0
# )
# b = portfolio.Portfolio(
#     ['AAPL', 'MSFT', 'INTC', 'IBM', 'CSCO', 'AMZN', 'DIS', 'MCD', 'NKE', 'HD'],
#     '2013-01-01',
#     '2023-01-01',
#     0.0,
#     1.0
# )
