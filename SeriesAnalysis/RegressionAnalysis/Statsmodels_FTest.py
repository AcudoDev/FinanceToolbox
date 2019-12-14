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

def Statsmodels_FTest(results, Explanatory, NumDecimal):
    """
    This function performs a FTest on the results of a statsmodels model.

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
        - FTest: str
            The FTest results
    """

    A = np.identity(len(results.params))
    A = A[1:, :]
    FTest = results.f_test(A)
    FTest = round(ss.f.cdf(FTest.fvalue, dfn=FTest.df_num, dfd=FTest.df_denom)[0][0], NumDecimal)

    return FTest
