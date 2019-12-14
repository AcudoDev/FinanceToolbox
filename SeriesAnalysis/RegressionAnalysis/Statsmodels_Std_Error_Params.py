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

def Statsmodels_Std_Error_Params(name, results, Explanatory, NumDecimal):
    """
    This function gives the Std Error of the of the Params from a statsmodels model results.

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
        - StErrParams: str
            The Std Error of the Params
    """

    StErrParams = results.bse
    StErrParams = [str(round(item, NumDecimal)) for item in StErrParams]
    for item in range(0, len(Explanatory.columns)):
        StErrParams[item + 1] = str(StErrParams[item + 1]) + ' ' + str(Explanatory.columns[item])
    StErrParams[0] = str(StErrParams[0])
    StErrParams = ', '.join(StErrParams)

    return StErrParams
