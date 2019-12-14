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

def AugmentedDickeyFuller(series, regression="ctt"):
    """
    This function do a Augmented Dickey Fuller test.

    Arguments:
    ----------
        - series: Pandas Series
            The series needed to perform the test
        - regression: str
            The regression type to use for the test

    Return:
    ----------
        - ADF: str
            The result of the test.
    """

    ADF = ts.adfuller(series, regression=regression)

    if ADF[0] < ADF[4]['1%']:
        ADF = 'Stationary 99%'
    elif ADF[0] < ADF[4]['5%']:
        ADF = 'Stationary 95%'
    elif ADF[0] < ADF[4]['10%']:
        ADF = 'Stationary 90%'
    else:
        ADF = 'Not Stationary'

    return ADF