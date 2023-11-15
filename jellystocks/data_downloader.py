# data_downloader.py

import yfinance as yf


def download_data(tickers, start_date, end_date):
    """
    Downloads historical 'Adj Close' data for the given tickers from Yahoo Finance.

    Parameters:
    tickers (list of str): Stock tickers.
    start_date (str): Start date in 'YYYY-MM-DD' format.
    end_date (str): End date in 'YYYY-MM-DD' format.

    Returns:
    pandas.DataFrame: Historical 'Adj Close' prices.
    """
    try:
        data = yf.download(tickers, start=start_date, end=end_date)['Adj Close']
        if data.empty:
            raise ValueError("No data fetched. Please check the tickers and date range.")
        return data[tickers]
    except Exception as e:
        raise ConnectionError(f"Failed to download data: {e}")
