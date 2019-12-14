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

import FinanceToolbox as FT

def HalfLife(series):
    """
    This function calculate the Halflife based on a series.
    """
    return round(-np.log(2))/FT.SeriesAnalysis.SpeedMeanReversion.SpeedMeanReversion(series)
