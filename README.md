[![License](https://img.shields.io/badge/license-BSD--2--Clause-blue.svg)](LICENSE)

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

## Origin of Code

This package includes modified code originally written by nhat truong. The original MATLAB code is subject to the following license:

```plaintext
Copyright (c) 2020, nhat truong
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions, and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions, and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

* Neither the name of nhat truong nor the names of its contributors may be
  used to endorse or promote products derived from this software without
  specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
```
You can find the original code [here](https://uk.mathworks.com/matlabcentral/fileexchange/78961-jellyfish-search-optimizer-js)

## License

This Python package, including the modifications, is licensed under the BSD-2-Clause License. See the [LICENSE](LICENSE) file for details.

```plaintext
BSD 2-Clause License

Copyright (c) 2023, Paul Barros Bessone
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions, and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions, and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

