from FinanceToolbox import imports
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


def FromYahooFinance(Symbol, Period):
    """
    This function retrieve the ohlcv data from Yahoo Finance for an asset in a given period.

    Arguments:
    ----------
        - Symbol:str
            The symbol of the asset
        - Period: str
            The period in past from today. i.e. 1mo, 1y

    Return:
        - HData: Pandas DatafFrame
            The historical data (ohlcv) for the asset and period given
    """

    Ticker = yf.Ticker(Symbol)
    HData = Ticker.history(period=Period)
    HData = HData.sort_index()

    return HData
