# portfolio.py
from tabulate import tabulate
from data_downloader import download_data
import pandas as pd
import numpy as np
import utils
from jellyfish_search import JellyfishOptimizer
from datetime import datetime


class Portfolio:
    """
    A class to represent and manage a stock portfolio.

    Attributes:
        tickers (list of str): A list of stock tickers in the portfolio.
        start_date (str): The start date for the data (format: 'YYYY-MM-DD').
        end_date (str): The end date for the data (format: 'YYYY-MM-DD').
        data (pd.DataFrame): DataFrame containing the historical stock data.
    """

    def __init__(self, tickers, start_date, end_date, lower_bound=0, upper_bound=1, risk_free_rate=0.02):
        """
        A class to represent and manage a stock portfolio.

        Parameters
        ----------
        tickers : list of str
            A list of stock tickers.
        start_date : str
            The start date for the data, formatted as 'YYYY-MM-DD'.
        end_date : str
            The end date for the data, formatted as 'YYYY-MM-DD'.
        lower_bound : float, optional
            The lower bound for portfolio weights. Default is 0.
        upper_bound : float, optional
            The upper bound for portfolio weights. Default is 1.
        risk_free_rate : float, optional
            The risk-free rate for portfolio evaluation. Default is 0.02.

        Attributes
        ----------
        tickers : list of str
            List of stock ticker symbols.
        start_date : str
            Start date for historical data.
        end_date : str
            End date for historical data.
        data : pandas.DataFrame or None
            DataFrame containing the historical stock data, None until data is fetched.
        risk_free_rate : float
            The risk-free rate used for portfolio evaluation.

        Notes
        -----
        The stock data is downloaded using Yahoo Finance API via the `yfinance` library.
        The date range must be valid, and the tickers must exist in the Yahoo Finance database.
        """

        # Validate inputs
        self.validate_inputs(tickers, start_date, end_date, lower_bound, upper_bound)

        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.data = None
        self._dimension = len(self.tickers)  # Directly assign dimension
        self.lower_bound = utils.construct_bounds_if_needed(self._dimension, lower_bound)
        self.upper_bound = utils.construct_bounds_if_needed(self._dimension, upper_bound)
        self._weights = None
        self._log_returns = None
        self._annual_cov_matrix = None
        self.risk_free_rate = risk_free_rate
        self.sharpe_ratio = None
        self.annualized_expected_returns = None
        self.portfolio_standard_deviation = None

        # Placeholder for data downloading logic
        self.download_data()
        self.calculate_equal_weights()
        self.calculate_log_returns()
        self.calculate_cov_matrix()
        self.calculate_portfolio_expected_returns()
        self.calculate_portfolio_standard_deviation()
        self.calculate_sharpe_ratio()

    def download_data(self):
        """
        Downloads historical 'Adj Close' data for the tickers in the portfolio.
        """
        self.data = download_data(self.tickers, self.start_date, self.end_date)

    def optimize(self, optimizer: JellyfishOptimizer, num_runs: int = 1):
        """
        Optimizes the portfolio using the Jellyfish Search optimization algorithm for multiple runs.

        Parameters
        ----------
        optimizer : JellyfishOptimizer
            An instance of the JellyfishOptimizer class.
        num_runs : int, optional
            Number of optimization runs, by default 1.

        Returns
        -------
        Tuple
            A tuple containing the best solution and its parameters across all runs,
            as well as a list of dictionaries representing all solutions found during each run.
        """
        best_global_solution = None
        best_global_cost = float('inf')  # Initialize with a large value
        best_global_iterations = 0
        best_global_evaluations = 0
        best_global_elapsed_time = 0
        best_global_cost_over_time = []
        all_solutions_history = []

        for run in range(num_runs):
            # Perform optimization for each run
            best_solution, best_cost, iterations, evaluations, elapsed_time, cost_over_time = optimizer.optimize(
                self._log_returns,
                self._annual_cov_matrix,
                self._dimension,
                self.lower_bound,
                self.upper_bound,
                self.risk_free_rate
            )

            # Update the best solution if the current run has a lower cost
            if best_cost < best_global_cost:
                best_global_solution = abs(best_solution)
                best_global_cost = best_cost
                best_global_iterations = iterations
                best_global_evaluations = evaluations
                best_global_elapsed_time = elapsed_time
                best_global_cost_over_time = cost_over_time

            # Save the current run's solution to the history as a dictionary
            current_run_dict = {
                'solution': abs(best_solution),
                'cost': best_cost,
                'iterations': iterations,
                'evaluations': evaluations,
                'elapsed_time': elapsed_time,
                'cost_over_time': cost_over_time
            }
            all_solutions_history.append(current_run_dict)

        # Update portfolio attributes with the best solution
        self._weights = best_global_solution
        self.sharpe_ratio = best_global_cost
        self.calculate_portfolio_expected_returns()
        self.calculate_portfolio_standard_deviation()

        return (best_global_solution, best_global_cost, best_global_iterations, best_global_evaluations,
                best_global_elapsed_time, best_global_cost_over_time, all_solutions_history)

    def display_portfolio(self):
        """
        Displays the portfolio in a user-friendly format.
        """
        # Code to display the portfolio, e.g., print a table of tickers and _weights
        pass

    # You can add more methods as needed for functionality like calculating returns,
    # risk metrics, etc.
    @staticmethod
    def validate_date(date_str):
        try:
            return datetime.strptime(date_str, "%Y-%m-%d")
        except ValueError:
            raise ValueError(f"Date '{date_str}' is not in the correct format (YYYY-MM-DD).")

    @staticmethod
    def validate_inputs(tickers, start_date, end_date, lower_bound, upper_bound):
        """
        Validates the inputs for the Portfolio constructor.

        Parameters
        ----------
        tickers : list of str
        start_date : str
        end_date : str

        Raises
        ------
        ValueError
            If the input validation fails.
            :param tickers:
            :param start_date:
            :param end_date:
            :param upper_bound:
            :param lower_bound:
        """
        if not isinstance(tickers, list) or len(tickers) < 2:
            raise ValueError("Tickers must be a list containing at least two items.")

        if not isinstance(tickers, list) or not all(isinstance(ticker, str) for ticker in tickers):
            raise ValueError("Tickers must be a list of strings.")

        start_date_obj = Portfolio.validate_date(start_date)
        end_date_obj = Portfolio.validate_date(end_date)

        if start_date_obj >= end_date_obj:
            raise ValueError("Start date must be before end date.")

        if lower_bound >= upper_bound:
            raise ValueError("Upper bound value must be greater than lower bound")

        if not tickers:
            raise ValueError("Tickers list cannot be empty.")

        if len(tickers) != len(set(tickers)):
            raise ValueError("Duplicate tickers detected in the tickers list.")

    def calculate_equal_weights(self):
        length = len(self.tickers)
        weights = np.array([1 / length] * length)
        self._weights = weights

    def calculate_log_returns(self):
        """
        Calculates and annualizes the log returns of the stock prices.
        """
        self._log_returns = np.log(self.data / self.data.shift(1)).dropna()

    def calculate_cov_matrix(self):
        """
        Calculates and annualizes the covariance matrix of the log returns.
        """
        self._annual_cov_matrix = self._log_returns.cov() * 252

    def calculate_portfolio_expected_returns(self):
        annualized_expected_return = utils.calculate_portfolio_expected_returns(self._weights, self._log_returns)
        self.annualized_expected_returns = annualized_expected_return

    def calculate_portfolio_standard_deviation(self):
        std = utils.calculate_portfolio_standard_deviation(abs(self._weights), self._annual_cov_matrix)
        self.portfolio_standard_deviation = std

    def display_optimization_results(self):
        ticker_weight_table = []
        metrics_table = []

        # Add portfolio composition to the ticker_weight_table
        for ticker, weight in zip(self.tickers, self._weights):
            ticker_weight_table.append([ticker, weight])
            # ticker_weight_table.append([ticker, f"{abs(weight) * 100:.4f}%"])

        # Add performance metrics to the metrics_table with percentage formatting
        metrics_table.append(["Optimized Sharpe Ratio", f"{abs(self.sharpe_ratio):.4f}"])
        metrics_table.append(["Annualized Expected Return", f"{self.annualized_expected_returns * 100:.4f}%"])
        metrics_table.append(["Annualized Volatility", f"{self.portfolio_standard_deviation * 100:.4f}%"])

        # Print the tables
        print("Portfolio Composition:")
        headers = ["Ticker", "Weight"]
        print(tabulate(ticker_weight_table, headers, tablefmt="simple"))

        print("\nPerformance Metrics:")
        headers = ["Metric", "Value"]
        print(tabulate(metrics_table, headers, tablefmt="simple"))

    def calculate_sharpe_ratio(self):
        # print(f'passing risk-free from portfolio: {self.risk_free_rate}')
        self.sharpe_ratio = utils.calculate_portfolio_sharpe_ratio(
            self._weights,
            self._log_returns,
            self._annual_cov_matrix,
            self.risk_free_rate
        )
