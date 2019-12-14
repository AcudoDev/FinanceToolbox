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

def Statsmodels_FittedValues(df, results, name):
    """
    This function is used to get the fitted values of a statsmodels model results

    Arguments:
    ----------
        - df: Pandas DataFrame
            The DataFrame where the values will be stored (inner merging on index)
        - name: str
            The name of the model (i.e OLS, GLSAR, ...).
        - results: statsmodels result object of the model
            The results object of the model

    Return:
    ----------
        - df: Pandas DataFrame
            The inputed DataFrame augmented of the new series
    """

    FittedValues = pd.DataFrame(results.predict())
    if name in ["RecursiveLS", "QuantReg"] or "GLM" in name:
        FittedValues = pd.DataFrame(
            FittedValues.iloc[:-1])  # ??? Why fitted values have more values than the original data series?

    if name == "AR":
        FittedValues.index = df.index[14:]
    elif name == "ARMA":
        FittedValues.iloc[15:].index = df.index
    elif name == "ARIMA":
        FittedValues.iloc[13:].index = df.index

    else:
        FittedValues.index = df.index
    FittedValues.rename(inplace=True, columns={FittedValues.columns[0]: name + " Fitted Values"})
    df = df.merge(FittedValues, how="inner", left_index=True, right_index=True)

    return df
