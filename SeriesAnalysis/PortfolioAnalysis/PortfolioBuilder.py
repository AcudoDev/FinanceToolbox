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

def PortfolioBuilder(HistoricalData, PortfolioComposition):
    """
    This function allow to calculate the portfolio value series

    Arguments:
    ----------
        - Historical Data: Dictionary of Pandas DataFrame
            Each Key is SheetName and the item associated is the Pandas Dataframe of the historical values

    Return:
    ----------
        - PortfolioValue: Pandas DataFrame
            Contain the Portfolio Value Series
    """
    numSheet = 1

    for SheetName in PortfolioComposition["SheetName"].unique().tolist():

        numLine = 1

        for index, row in PortfolioComposition[PortfolioComposition["SheetName"] == SheetName].iterrows():

            Ticker = row["Ticker"]
            Weight = row["Weight"]

            HistoricalData[SheetName][Ticker] = pd.to_numeric(HistoricalData[SheetName][Ticker], errors="coerce")
            HistoricalData[SheetName][Ticker].dropna(inplace=True)
            if len(HistoricalData[SheetName][Ticker]) > 0:

                if numLine == 1:
                    Data = HistoricalData[SheetName][Ticker] * Weight
                else:
                    Data = pd.DataFrame(Data).merge(HistoricalData[SheetName][Ticker] * Weight, how="inner",
                                                left_index=True, right_index=True)

            else:
                return "Error"

            numLine = numLine + 1

        Data[SheetName + "_" + "Portfolio Value"] = Data.sum(axis=1)

        if numSheet == 1:
            Portfolio = Data
        else:
            Portfolio = pd.DataFrame(Portfolio).merge(Data, how="inner", left_index=True, right_index=True)

        numSheet = numSheet + 1

    return Portfolio
