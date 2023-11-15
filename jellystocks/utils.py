import numpy as np


def get_risk_free():
    # implement an automated calculation of risk-free
    return 0.02


def construct_bounds_if_needed(dimension, bound):
    # If bound is an integer, create a list of that length
    if isinstance(bound, (int, float)):
        return np.array([bound] * dimension)

    # If bound is a list, ensure it matches the dimension
    elif isinstance(bound, list):
        if len(bound) == dimension:
            return np.array(bound)
        else:
            raise ValueError(f"Length of bound list ({len(bound)}) does not match dimension of portfolio "
                             f"({dimension}).")

    # Handle None or other types
    else:
        raise ValueError("Bound must be a numeric value or a list.")


def calculate_portfolio_expected_returns(weights, log_returns):
    # print(f'log returns from inside fn: {log_returns}')

    mean_returns = log_returns.mean(axis=0)
    # print(f'mean returns from inside fn: {mean_returns}')
    # print(f'weights inside fn: {weights}')
    daily_expected_return = np.dot(weights, mean_returns)
    # print(f'daily exp returns from inside fn: {daily_expected_return}')

    annualized_expected_return = daily_expected_return * 252
    # print(f'annualized returns from inside fn: {annualized_expected_return}')
    return annualized_expected_return

# def calculate_portfolio_expected_returns(weights, log_returns):
#     """
#     Calculates the annualized expected returns using log returns.
#
#     Parameters:
#     weights (array): Array of weights for each stock in the portfolio.
#     log_returns (DataFrame): DataFrame containing the log returns of the stocks.
#
#     Returns:
#     float: Annualized expected return of the portfolio.
#     """
#     # Calculate the sum of weighted log returns
#     weighted_log_returns = log_returns.dot(weights)
#
#     # Sum up the weighted log returns to get the total log return
#     total_log_return = weighted_log_returns.sum()
#
#     # Calculate the annualized return using the exponential function
#     # (252 is the number of trading days in a year)
#     annualized_expected_return = np.exp(total_log_return * (252 / len(log_returns))) - 1
#
#     return annualized_expected_return


def calculate_portfolio_standard_deviation(weights, cov_matrix):
    std = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
    return std


def calculate_portfolio_sharpe_ratio(weights, log_returns, cov_matrix,
                                     risk_free=get_risk_free()):

    print(f'utils sharpe being used')
    print(f'risk free: {risk_free}')
    print(f'weights: {weights}')
    port_return = calculate_portfolio_expected_returns(weights, log_returns)
    port_std_dev = calculate_portfolio_standard_deviation(weights, cov_matrix)
    sharpe = (port_return - risk_free) / port_std_dev

    return sharpe
