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

def KFLinearRegression(Independent, Explanatory, R=0.1, Q=0.01, Pfactor=1.):
    dim_x = 1 + len(Explanatory.columns)  # alpha, béta : y = alpha + beta1 * x1 + beta2 * x2 + ... + betaN * xN
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
    dt = 1
    filter.Q = Q_discrete_white_noise(dim_x, dt, Q)

    ParamsResults = []
    CovMatrix = []
    Variance = []
    ModelResiduals = []

    for i in range(0, len(Independent)):
        z = np.array(Independent.iloc[i, :].values.tolist())

        # Measurement function
        HArray = [1]
        for j in range(0, len(Explanatory.columns)):
            HArray.append(Explanatory[Explanatory.columns[j]][i - 1])
        HArray = np.array([HArray])

        filter.H = HArray

        filter.predict()
        filter.update(z)

        x = filter.x.tolist()
        x = [item[0] for item in x]
        ParamsResults.append(x)

        P = filter.P
        CovMatrix.append(P)
        Variance.append(np.diagonal(P))

        y = filter.y.tolist()
        y = [item[0] for item in y]
        ModelResiduals.append(y)

    ParamsResults = pd.DataFrame(ParamsResults)
    NameColumn = ["Intercept"]
    for k in range(0, len(Explanatory.columns)):
        NameColumn.append("Coefficient - " + Explanatory.columns[k])

    for j in range(0, len(ParamsResults.columns)):
        ParamsResults = ParamsResults.rename(columns={ParamsResults.columns[j]: "" + NameColumn[j]})

    Variance = pd.DataFrame(Variance)
    for j in range(0, len(Variance.columns)):
        Variance = Variance.rename(columns={Variance.columns[j]: "Var - " + NameColumn[j]})

    ModelResiduals = pd.DataFrame(ModelResiduals)
    ModelResiduals = ModelResiduals.rename(columns={ModelResiduals.columns[0]: "KF Residuals"})

    Results = ParamsResults.merge(Variance, left_index=True, right_index=True)
    Results = Results.merge(ModelResiduals, left_index=True, right_index=True)
    Results.index = Independent.index
    return Results
