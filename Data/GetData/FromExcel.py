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


def FromExcel(file, sheet):
    '''
    This function is used to import an excel sheet to a Pandas DataFrame
    ------------
    Arguments:
        - File:str
                The absolute path of the file
        - sheet:str
                The sheet name to import

    Return:
        A Pandas Dataframe containing the data of the excel sheet
    '''

    WB_Input = pyxl.load_workbook(file, data_only=True, keep_vba=True)
    content = WB_Input[sheet].values
    columns = next(content)[0:]
    WB_Input.close()

    return pd.DataFrame(content, columns=columns)
