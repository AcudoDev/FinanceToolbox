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

from filterpy.kalman import KalmanFilter


def KF_x1z1(df, R=1, Q=0.01, Pfactor=1.):
    dim_x = 1  # alpha, béta : y = alpha + beta1 * x1 + beta2 * x2 + ... + betaN * xN
    dim_z = 1  # independent variable

    filter = KalmanFilter(dim_x=dim_x, dim_z=dim_z)

    # Initial state
    filter.x = np.zeros((dim_x, 1), float)

    # state transition matrix
    filter.F = np.eye(dim_x, dtype=float)

    # covariance matrix
    filter.P *= Pfactor

    # state uncertainty
    filter.R = R

    # process uncertainty
    filter.Q = Q

    ParamsResults = []
    CovMatrix = []
    Variance = []
    ModelResiduals = []

    for i in range(0, len(df)):
        z = np.array(df.iloc[i, :].values.tolist())

        # Measurement function
        HArray = [1]
        HArray = np.array([HArray])

        filter.H = HArray

        filter.predict()
        filter.update(z)

        x = filter.x.tolist()
        ParamsResults.append(x)

        P = filter.P
        CovMatrix.append(P)
        Variance.append(np.diagonal(P))

        y = filter.y.tolist()
        y = [item[0] for item in y]
        ModelResiduals.append(y)

    ParamsResults = [item[0] for item in ParamsResults]
    ParamsResults = pd.DataFrame(ParamsResults)
    ParamsResults = ParamsResults.rename(columns={ParamsResults.columns[0]: "KF " + df.columns[0]})

    Variance = pd.DataFrame(Variance)
    Variance = Variance.rename(columns={Variance.columns[0]: "KF " + df.columns[0] + " Variance"})

    ModelResiduals = pd.DataFrame(ModelResiduals)
    ModelResiduals = ModelResiduals.rename(columns={ModelResiduals.columns[0]: "KF " + df.columns[0] + " Residuals"})

    Results = ParamsResults.merge(Variance, left_index=True, right_index=True)
    Results = Results.merge(ModelResiduals, left_index=True, right_index=True)

    Results.index = df.index

    return Results
