 
# Investment

Investment is a Python library for dealing with stock market investment.

## Usage

```python
from investment import Portfolio

budget = 2000
tickers = [ 'GBMINTBO.MX', 'FMTY14.MX', 'GLD','VTI.MX', 'CEMEXCPO.MX', "IVVPESO.MX",] # list(acciones.values())
tickers_change = [x for x in tickers if ".MX" not in x]

port = Portfolio( budget, tickers, start='2018-01-01', end=None, interval="d", tickers_change=tickers_change, currency_change=["MXN=X"])

port.get_data() 
port.data # returns Data frame with prices
port.get_best_portfolio(100000,risk_free=0,returns_periods=30, budget=12000)
port.best_portfolio # dictionary with best portfolios (sharpe, sortino, min colatility)
```
```python

port.simulation # dictionary with all simulations
boll = port.bollinger_est() #DataFrame with bollinger bands and counter
```
## Contributing
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.
