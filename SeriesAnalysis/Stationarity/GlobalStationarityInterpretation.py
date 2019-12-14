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

def GlobalStationarityInterpretation(ADF, KPSS):
    """
    This function interpret the result of the ADF and KPSS for the stationarity of the series.

    Arguments:
    ----------
        - ADF: str
            The output of the AugmentedDickeyFuller function
        - KPSS: str
            The output of the KPSS_Test function

    Return:
    ----------
        - StationaryResult: str
            The Stationary Result / Type
    """

    if ADF == "Not Stationary" and KPSS == "Not Stationary":
        StationaryResult = "Not Stationary"

    elif "%" in ADF and "%" in KPSS:  # both stationary..
        StationaryResult = "Strict Stationary"

    elif "%" in ADF and KPSS == "Not Stationary":  # only ADF stationary
        StationaryResult = "Difference Stationary"
    elif ADF == "Not Stationary" and "%" in KPSS:  # only KPSS stationary
        StationaryResult = "Trend Stationary"

    return StationaryResult
