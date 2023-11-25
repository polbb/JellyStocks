from setuptools import setup

# Read dependencies from requirements.txt
with open('requirements.txt') as f:
    dependencies = f.read().splitlines()

setup(
    name='JellyStocks',
    version='1.0.4',
    packages=['jellystocks'],  # to exclude other dir that can be auto recognized as packages
    install_requires=dependencies,
    license='BSD-2-Clause',
    long_description="""
JellyStock
==========

JellyStock is a Python library for optimizing stock portfolios using the Jellyfish Search optimization algorithm. It 
allows you to create and manage portfolios of stocks, download historical stock data, and find the optimal portfolio 
composition that maximizes the Sharpe ratio.

Features
--------

- **Portfolio Management**: Create and manage portfolios with a list of stock tickers, specifying the start and end 
dates for historical data.

- **Historical Data**: Download historical 'Adj Close' data for a list of stock tickers from Yahoo Finance.

- **Portfolio Optimization**: Use the Jellyfish Search optimization algorithm to find the optimal portfolio weights 
that maximize the Sharpe ratio.

- **Risk Analysis**: Calculate risk metrics such as expected returns, volatility, and Sharpe ratio for your portfolio.

- **User-Friendly**: Provides user-friendly methods for displaying portfolio information and optimization results.
""",
    long_description_content_type='text/markdown'
)
