import pandas as pd
import numpy as np

import yfinance as yf
from sklearn.linear_model import LinearRegression
import statsmodels
import statsmodels.api as sm
import statsmodels.tsa.stattools as ts

import datetime

import scipy.stats
import math
import openpyxl as pyxl
from scipy import signal
from scipy import stats as ss
import statistics

from finta import TA
from filterpy.kalman import KalmanFilter
from filterpy.common import Q_discrete_white_noise

import pandas_ta as ta
from pingouin import gzscore


def GaussianRandomStockPrice(mu, sigma, n, end, freq, S0=100):
    """
    This function randomly creates a stock price series bases on gaussian probabilities.

    Arguments:
    ----------
        - mu: float
            The mean parameter
        - sigma: float
            The standard déviation parameter
        - n: int
            Number of periods
        - end: datetime date
            The last date of thé series
        - freq: pandas frequency string
            The frequency of thé dataseries:
                - "D": days
                - "min": minutes
                - "s": seconds
        - S0: float
            The first stock price

    Return:
    ----------
        - RStock: Pandas DataFrame
            Contains thé datetime as index and thé random stock prices in a column

    """

    RStock = np.random.normal(mu, sigma, n).astype("float")
    RStock = pd.DataFrame(RStock)
    RStock.rename(inplace=True, columns={RStock.columns[0]: "Return"})
    RStock["Price"] = ((1 + RStock["Return"]).cumprod()) * S0
    times = pd.date_range(end=end, freq=freq, periods=n)

    RStock.index = times
    RStock = pd.DataFrame(RStock["Price"])

    return RStock
