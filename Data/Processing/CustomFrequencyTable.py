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

def CustomFrequencyTable(Series, step, center=0):
    """
    This function creates a custom frequency table
    ----------
    Arguments:
        - Series: Pandas Series
            The series to use for creating the custom frequency table
        - step: float
            The size of each class
        - multiplicator: float
            Used to change how the labels look like in a graph. A multiplicator of 1000 will multiply by 1000 the left and right element of a class (but it's only a "visual" change)
        - center: float
            The center value of the central class
        - rounding: int
            The rounding of the class labels

    Return:
        FreqTable: Pandas DataFrame
            A dataframe containing all classes and their frequency
    """

    rounding = len(str(step).split(".")[-1]) + 1

    min = Series.min()
    max = Series.max()

    ClassList = []

    i = center - step / 2
    while i >= min:
        NewClass = [i, i - step]
        ClassList.append(NewClass)
        i = i - step

    i = center + step / 2
    while i <= max:
        NewClass = [i, i + step]
        ClassList.append(NewClass)
        i = i + step

    ClassList[len(ClassList) - 1][1] = ClassList[len(ClassList) - 1][
                                           1] + 0.00000000000000001  # Each class will take all values >= to the left leg
    # and < (strictly) to the right leg.The right leg for the last class is excluded whereas it corresponds to the MAX. So, we increase a little bit the class size
    # to take into account the MAX value inside the last class.

    ClassList.append([(center - step / 2), (center + step / 2)])

    for i in range(0, len(ClassList)):
        for j in range(0, len(ClassList[i])):
            ClassList[i][j] = round(ClassList[i][j], rounding)
        if ClassList[i][0] < ClassList[i][1]:
            leftel = ClassList[i][0]
            rightel = ClassList[i][1]
        else:
            leftel = ClassList[i][1]
            rightel = ClassList[i][0]
        ClassList[i] = pd.Interval(left=leftel, right=rightel)

    CountList = []

    for item in ClassList:
        Count = Series[(Series >= item.left) & (Series < item.right)].count()
        CountList.append(Count)

    CountList = [item / sum(CountList) for item in CountList]

    FreqTable = pd.concat([pd.Series(ClassList), pd.Series(CountList)], axis=1)
    FreqTable.index = FreqTable[FreqTable.columns[0]]
    FreqTable = FreqTable.drop(FreqTable.columns[0], axis=1)
    FreqTable = FreqTable.sort_index()

    return FreqTable
