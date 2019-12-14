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

def Pandas_DataFrame_to_win32comSheet(df, win32com_SheetObject, StartRow=1, StartCol=1, data_type="Normal",
                                      excel_table=True, autofit=True):
    """
    This function write a Pandas DataFrame to an excel sheet with win32com

    Arguments:
    ----------
        - df: Pandas DataFrame
            The dataframe to write
        - win32com_SheetObject: a win32com Excel Sheet Object
            The win32com Excel sheet where to write the dataframe
        - StartRow: int
            The starting row number
        - StartCol: int
            The starting column number
        - data_type: str
            The type of the data:
               - Normal: no adte column
                - WithDate: have a date column as index (first column)
        - excel_table: boolean
            Convert the results into an excel table if True
    Return:
    ----------
        Nothing

    """
    if data_type == "Normal":
        # Values
        win32com_SheetObject.Range(win32com_SheetObject.Cells(StartRow + 1, StartCol),
                                   win32com_SheetObject.Cells(StartRow + 1 + len(df.index) - 1,
                                                              StartCol + len(df.columns) - 1)
                                   ).Value = df.values.tolist()
        # Headers (columns names)
        win32com_SheetObject.Range(win32com_SheetObject.Cells(StartRow, StartCol), win32com_SheetObject.Cells(StartRow,
                                                                                                              StartCol + len(
                                                                                                                  df.columns) - 1)).Value = df.columns
    elif data_type == "WithDate":
        # Values from the 2nd column (excluding the first column, the date column)
        win32com_SheetObject.Range(win32com_SheetObject.Cells(StartRow + 1, StartCol + 1),
                                   win32com_SheetObject.Cells(StartRow + 1 + len(df.index) - 1,
                                                              StartCol + 1 + len(df.columns) - 1)
                                   ).Value = df.values.tolist()

        # Headers
        win32com_SheetObject.Range(win32com_SheetObject.Cells(StartRow, StartCol + 1),
                                   win32com_SheetObject.Cells(StartRow,
                                                              StartCol + 1 + len(df.columns) - 1)).Value = df.columns

        # Values, date column
        win32com_SheetObject.Range(win32com_SheetObject.Cells(StartRow + 1, StartCol),
                                   win32com_SheetObject.Cells(StartRow + 1 + len(df.index) - 1,
                                                              StartCol)).Value = df.index.strftime("%Y-%m-%d").tolist()

        # for i in range(0, len(Data.index)):
        #    wsTemp.Cells(StartRow + i, StartCol - 1).Value = Data.index.strftime("%Y-%m-%d").tolist()[i]

    if data_type == "Normal":
        rng = win32com_SheetObject.Range(win32com_SheetObject.Cells(StartRow, StartCol),
                                         win32com_SheetObject.Cells(StartRow + len(df.index) - 1,
                                                                    StartCol + len(df.columns) - 1))
    elif data_type == "WithDate":
        rng = win32com_SheetObject.Range(win32com_SheetObject.Cells(StartRow, StartCol),
                                         win32com_SheetObject.Cells(StartRow + len(df.index) - 1,
                                                                    StartCol + 1 + len(df.columns) - 1))

    if excel_table == True:
        win32com_SheetObject.ListObjects.Add(1, rng, XlListObjectHasHeaders=1)

    if autofit == True:
        rng.EntireColumn.AutoFit()

    return