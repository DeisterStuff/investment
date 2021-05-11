import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
from functools import reduce
import time
from datetime import datetime
from portfolio_funcs import get_weights, simulate_portfolios, factible_weights
from trade_utils import *


class Portfolio:
    def __init__(self, budget, tickers, start='2018-01-01', end=None, interval="d", tickers_change=[], currency_change=["MXN=X"]):
        """Get stock pices, currency, calculate best portfolio for investing depending on your  budget

        Args:
            budget (float): Total Budget to invest (presupuesto)
            tickers (list[str]): list of every ticker (yahoo finance)
            start (str, optional): start date to request the data. Defaults to '2018-01-01'.
            end (str, optional): start date to request the data, if None end date is latest date.  Defaults to None.
            interval (str, optional): interval of requesting data, "d" means every day. Defaults to "d".
            tickers_change (list, optional): List of tickers which want to change to a currency. Defaults to [].
            currency_change (list, optional): Ticker of the currency change. Defaults to ["MXN=X"].
        """
        self.budget = budget
        self.tickers = tickers
        self.start = start
        self.end = end
        self.interval = interval
        self.tickers_change = tickers_change
        self.currency_change = currency_change

    def get_data(self):
        """Gets Data from Yahoo Finance

        Returns:
            pandas.DataFrame: dataframe with stock data
        """
        data_list = [
            pdr.get_data_yahoo([l],
                               start=self.start,
                               end=self.end,
                               interval=self.interval) for l in self.tickers
        ]

        self.data = self.merge_list(data_list)

        self.data = self.data.groupby(by='Date').mean()
        self.adj_close = self.data['Adj Close'].fillna(method='ffill').dropna()
        return self.data

    def change_currency(self):
        """Transform the prices to the selected currency
        """
        if len(self.tickers_change) > 0:
            change = pdr.get_data_yahoo(self.currency_change,
                                        start=self.start,
                                        end=self.end,
                                        interval=self.interval)
            new = self.change_currency_(
                self.adj_close[self.tickers_change], change['Adj Close'])
            self.adj_close.drop(columns=self.tickers_change, inplace=True)
            self.adj_close = pd.merge(
                self.adj_close, new, how="left", on="Date")
            self.adj_close.fillna(method='ffill', inplace=True)
            self.adj_close.dropna(inplace=True)
        else:
            pass

    def get_best_portfolio(self, n_portfolios=10000, risk_free=0, returns_periods=180, budget=None):
        """simulate n portfolios and evaluate the risk and profit associated on each one to calculate the best option

        Args:
            n_portfolios (int, optional): Number of simulated portfolios. Defaults to 10000.
            risk_free (int, optional): risk free profit (tipically treassure bonds). Defaults to 0.
            returns_periods (int, optional): number of periods to take on the calculation. Defaults to 180.
            budget (float, optional): Availiable budget. Defaults to None.

        Returns:
            dict: dictionary with metrics and number of stocks
                    """
        if budget == None:
            budget = self.budget

        n_assets = len(self.tickers)
        assets = self.adj_close.tail(1)

        weights = get_weights(
            n_assets=n_assets, n_portfolios=n_portfolios, sell=False)
        self.weights = factible_weights(budget, weights, assets.values)
        returns = self.adj_close.pct_change().dropna()
        returns = returns.tail(returns_periods)
        self.simulation = simulate_portfolios(
            self.weights, returns, risk_free=risk_free, returns_periods=returns_periods, sharpe=True, sortino=True)
        sharpe_index = self.simulation['sharpe'].argmax()
        sortino_index = self.simulation['sortino'].argmax()
        min_vol_index = self.simulation['volatility'].argmin()
        sharpe_weights = self.weights[sharpe_index]
        sortino_weights = self.weights[sortino_index]
        min_vol_weights = self.weights[min_vol_index]

        self.best_portfolio = {
            "sharpe": {
                "weigths": sharpe_weights,
                "n_buys": ((sharpe_weights*budget)/assets).round(),
                "budget": (((sharpe_weights*budget)/assets).round()*assets).values.sum(),
                "return": self.simulation['returns'][sharpe_index],
                "volatility": self.simulation['volatility'][sharpe_index],
                "ratio": self.simulation['sharpe'][sharpe_index],
            },
            "sortino": {
                "weigths": sortino_weights,
                "n_buys": ((sortino_weights*budget)/assets).round(),
                "budget": (((sortino_weights*budget)/assets).round()*assets).values.sum(),
                "return": self.simulation['returns'][sortino_index],
                "volatility": self.simulation['volatility'][sortino_index],
                "ratio": self.simulation['sortino'][sortino_index],
            },
            "min_vol": {
                "weigths": min_vol_weights,
                "n_buys": ((min_vol_weights*budget)/assets).round(),
                "budget": (((min_vol_weights*budget)/assets).round()*assets).values.sum(),
                "return": self.simulation['returns'][min_vol_index],
                "volatility": self.simulation['volatility'][min_vol_index],
                "ratio": self.simulation['sharpe'][min_vol_index],
            }
        }

        return self.best_portfolio

    def bollinger_est(self):
        """Creates df with Bollinger bands and a counter when the bands touch each other

        Returns:
            pandas.DataFrame: Data frame with bollinger bands 
        """
        data = self.adj_close
        df_list = []
        for tick in self.tickers:
            df = bollinger_bands(work_df=data, column=tick, period=20)
            # print(df)
            # Contador de bellinger
            # 1 vender -1 comprar
            contador = df[[f"close_{tick}", f"upper_{tick}", f"lower_{tick}"]].apply(
                lambda x: 1 if x[f"close_{tick}"] >= x[f'upper_{tick}'] else (-1 if x[f"close_{tick}"] <= x[f'lower_{tick}'] else 0), axis=1)
            df[f'cont_{tick}'] = (((contador.pct_change()*contador.shift(1)).fillna(
                0)).replace(0, np.nan).fillna(method='ffill') == 1).astype(int)
            df_list.append(df)
        data = self.merge_list(df_list)
        return data

    @staticmethod
    def get_new_data(tickers, interval, columns):
        """get new data from the tickers specified

        Args:
            tickers (list): list with stocks tickers
            interval (str): interval to request the data (like day,week,etc)
            columns (list): clumns wanted in the request

        Returns:
            pandas.DataFrame: [description]
        """
        today = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d")
        new_list = []
        for l in tickers:
            try:
                df = pdr.get_data_yahoo([l],
                                        start=today,
                                        end=None,
                                        interval=interval)
            except:
                # c = self.data.columns.get_level_values(0).unique().values
                df = pd.DataFrame(columns=pd.MultiIndex.from_tuples(
                    [(x, l) for x in columns]))

            new_list.append(df)
        new = reduce(lambda left, right: pd.merge(left, right,
                                                  left_index=True,
                                                  right_index=True,
                                                  how='outer'),
                     new_list)
        return new

    @staticmethod
    def merge_list(data_list):
        """list of data frames merged into one DataFrame

        Args:
            data_list (List[DataFrame]): list of data frames

        Returns:
            pandas.DataFrame: merged DataFrame
        """
        data = reduce(lambda left, right: pd.merge(left, right,
                                                   left_index=True,
                                                   right_index=True,
                                                   how='outer'),
                      data_list)
        return data

    @staticmethod
    def change_currency_(data, change):
        new = pd.merge(change, data, how="inner", on='Date')
        new = pd.DataFrame(new[change.columns].values.reshape(-1, 1)*new[data.columns].values,
                           columns=data.columns,
                           index=new.index)
        return new
