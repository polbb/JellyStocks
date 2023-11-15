The link below is a webapp for PO:
https://www.portfoliovisualizer.com/optimize-portfolio


portfilio class:
    - assests, weights, returns dataframe, cov martrix

    fn: download yahoo data
    create df logs
    create cov matrx


optimizer class:
    - population number, internal parameters???

    arguments: portfolio class


   Better to just use optimization as a method for portfolio class?



JS algo fn needs to either take the dfs with log ret and cov mx or calculate it itself.

if I have a PO object that contains all of that as attributes???

====================================================================================

JellyStocks/
│
├── jellystocks/
│   ├── __init__.py
│   ├── portfolio.py
│   ├── jellyfish_search.py
│   ├── cost_functions.py
│   ├── data_downloader.py
│   └── utils.py
│
├── tests/
│   ├── __init__.py
│   ├── test_portfolio.py
│   ├── test_jellyfish_search.py
│   └── test_data_downloader.py
│
├── docs/
│   ├── index.md
│   └── usage.md
│
├── examples/
│   └── example_usage.py
│
├── .gitignore
├── setup.py
├── README.md
├── LICENSE
└── requirements.txt



mean returns from inside fn: AAPL    -0.001325
GOOGL   -0.001987
IBM      0.000334
dtype: float64
weights inside fn: [0. 0. 1.]
daily exp returns from inside fn: 0.0003337377213083952
annualized returns from inside fn: 0.08410190576971559
Optimized Portfolio Composition:
AAPL: 0.0
IBM: 0.0
GOOGL: 1.0
Performance Metrics:
Optimized Sharpe Ratio: 0.2694
Annualized Expected Return: 8.4102%
Annualized Volatility: 23.7949%