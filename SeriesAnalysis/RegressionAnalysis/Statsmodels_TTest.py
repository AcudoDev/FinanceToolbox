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

def Statsmodels_TTest(results, Explanatory, NumDecimal):
    """
    This function performs a T Test on the results of a statsmodels model.

    Arguments:
    ----------
        - results: statsmodels result object of the model
            The results object of the model
        - Explanatory: Pandas DataFrame
            The DataFrame with the explanatory series
        - NumDecimal: int
            Number of decimals for the numbers calculated

    Return:
    ----------
        - TTest: str
            The TTest results
    """

    TTest = []
    for item in results.t_test(np.eye(len(results.params))).tvalue:
        TTest.append(ss.t.cdf(item, results.df_model))
    TTest = [str(round(item, NumDecimal)) for item in TTest]
    for item in range(0, len(Explanatory.columns)):
        TTest[item + 1] = str(TTest[item + 1]) + ' ' + str(Explanatory.columns[item])
    TTest[0] = str(TTest[0])
    TTest = ', '.join(TTest)

    return TTest
