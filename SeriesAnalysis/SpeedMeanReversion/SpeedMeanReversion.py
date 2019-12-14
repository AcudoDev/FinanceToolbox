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


def SpeedMeanReversion(series):
    """
    This function calculate the speed mean reversion.

    Arguments:
    ----------
        - series: Pandas Series
            The series needed to calculate the speed mean reversion

    Return:
    ----------
        - SpeedMR: float
            The speed mean reversion of the series.

    """

    Data_lag = series.shift(1).reset_index(drop=True)
    Data_lag = Data_lag.dropna()
    Data_returns = series.reset_index(drop=True) - Data_lag
    Data_returns = Data_returns.dropna()

    Data_lag2 = sm.add_constant(Data_lag)
    model = sm.OLS(Data_returns, Data_lag2)
    results = model.fit()
    SpeedMR = results.params[1]

    return SpeedMR
