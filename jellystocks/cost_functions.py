import numpy as np
import utils


def negative_portfolio_sharpe_ratio(weights, lower_bound, upper_bound, log_returns, cov_matrix,
                                    risk_free=utils.get_risk_free()):
    """
    Calculate the negative Sharpe ratio of a portfolio with penalties for weight deviations.

    Parameters
    ----------
    weights : numpy.ndarray
        Portfolio weights.
    lower_bound : numpy.ndarray
        Lower bounds for portfolio weights.
    upper_bound : numpy.ndarray
        Upper bounds for portfolio weights.
    log_returns : numpy.ndarray
        Logarithmic returns of the portfolio assets.
    cov_matrix : numpy.ndarray
        Covariance matrix of the portfolio assets.
    risk_free : float, optional
        Risk-free rate for the optimization. Default is the risk-free rate obtained from the utility function.

    Returns
    -------
    float
        Negative Sharpe ratio with penalties for weight deviations.
    """

    port_return = utils.calculate_portfolio_expected_returns(weights, log_returns)
    port_std_dev = utils.calculate_portfolio_standard_deviation(weights, cov_matrix)
    sharpe = -(port_return - risk_free) / port_std_dev
    # Add the penalties to the Sharpe ratio
    return sharpe + weight_penalty(weights) + boundary_penalty(weights, lower_bound, upper_bound)


def weight_penalty(weights):
    """
    Calculate the penalty for the sum of weights not equalling 1.

    Parameters
    ----------
    weights : numpy.ndarray
        Portfolio weights.

    Returns
    -------
    float
        Weight penalty.
    """
    lambda_ = 10  # Penalty factor, adjust this as needed
    return lambda_ * (np.sum(weights) - 1) ** 2


def boundary_penalty(weights, lower_bounds, upper_bounds):
    """
    Calculate the penalty for weights out of bounds.

    Parameters
    ----------
    weights : numpy.ndarray
        Portfolio weights.
    lower_bounds : numpy.ndarray
        Lower bounds for portfolio weights.
    upper_bounds : numpy.ndarray
        Upper bounds for portfolio weights.

    Returns
    -------
    float
        Boundary penalty.
    """
    lambda_ = 10
    penalty = 0

    below_bounds = weights < lower_bounds
    above_bounds = weights > upper_bounds

    penalty += lambda_ * np.sum(lower_bounds[below_bounds] - weights[below_bounds])
    penalty += lambda_ * np.sum(weights[above_bounds] - upper_bounds[above_bounds])
    return penalty


COST_FUNCTIONS = {
    'sharpe': negative_portfolio_sharpe_ratio
    # Add more mappings as needed
}
