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

def KPSS(series, regression="c"):
    """
    This function performs a KPSS test. Test if the data are trend-stationary or not.

    Arguments:
    ----------
        - series: Pandas Series
            The series to test

    Return:
        - KPSS: str
            The result of the test
    """
    KPSS = ts.kpss(series, regression=regression)
    if KPSS[0] < KPSS[3]['1%']:
        KPSS = 'Stationary 99%'
    elif KPSS[0] < KPSS[3]['5%']:
        KPSS = 'Stationary 95%'
    elif KPSS[0] < KPSS[3]['2.5%']:
        KPSS = 'Stationary 97.5%'
    elif KPSS[0] < KPSS[3]['10%']:
        KPSS = 'Stationary 90%'
    else:
        KPSS = 'Not Stationary'

    return KPSS
