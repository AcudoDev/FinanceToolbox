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

def DF_Treatment(df, dropna=False, set_index=False, set_index_name=None, sort_index=False,
                            convert_index_format=None):
    '''
    Basic Treatment for a DataFrame. Useful for other functions.
    ----------
    Arguments:
        - df: Pandas DataFrame
            The dataframe to treat.
        - dropna: True or False
            Avoid having NaNs
        - set_index: True or False
            Does an index must be set?
        - set_index_name: str
            The name of column to set as an index
        - sort_index: True or False
            If True, sort the index
        - convert_index_format:str
            Define how the index will be formatted. Possible format:
                - date_daily: %Y-%m-%d

    Return:
        The post-treatment Pandas DataFrame

    '''

    if dropna == True:
        df.dropna(inplace=True)

    if set_index == True:
        df.set_index(set_index_name, inplace=True)

    if sort_index == True:
        df.sort_index(inplace=True)

    if convert_index_format == "date_daily":
        df.index = df.index.strftime('%Y-%m-%d')

    return df