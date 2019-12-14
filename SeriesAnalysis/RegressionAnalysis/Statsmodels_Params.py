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

def Statsmodels_Params(name, results, Explanatory, NumDecimal):
    """
    This function gives the Params of a statsmodels model results.

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
        - Params: str
            The Params of the statsmodel model results
    """
    if name == "Holt Winter’s Exponential Smoothing":
        ResultsParams = results.params
        # ResultsParams = [round(item, NumDecimal) for item in ResultsParams]

        # for item in range(0, len(Explanatory.columns)):
        # ResultsParams[item+1] = str(ResultsParams[item+1]) + ' ' + str(Explanatory.columns[item])

        # ResultsParams[0] = str(ResultsParams[0])
        # ResultsParams = ', '.join(ResultsParams)
    elif "AR" in name:
        ResultsParams = results.params
        ResultsParams = [round(item, NumDecimal) for item in ResultsParams]

        for item in range(0, len(Explanatory.columns)):
            ResultsParams[item + 1] = str(ResultsParams[item + 1]) + ' ' + str(Explanatory.columns[item])

        ResultsParams[0] = str(ResultsParams[0])
        # ResultsParams = ', '.join(ResultsParams)

    else:
        ResultsParams = results.params
        ResultsParams = [round(item, NumDecimal) for item in ResultsParams]

        for item in range(0, len(Explanatory.columns)):

            ResultsParams[item + 1] = str(ResultsParams[item + 1]) + ' ' + str(Explanatory.columns[item])

        ResultsParams[0] = str(ResultsParams[0])
        ResultsParams = ', '.join(ResultsParams)

    return ResultsParams
