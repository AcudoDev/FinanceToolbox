from .Statsmodels_Params import *
from .Statsmodels_Std_Error_Params import *
from .Statsmodels_PValues import *
from .Statsmodels_FTest import *
from .Statsmodels_TTest import *

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

def Statsmodels_Regression_All_OneValueIndicators(OneValueIndicators, name, results, Explanatory, NumDecimal):
    """
    This function is used to calculate all common one-value indicators for linear regression with statsmodels

    Arguments:
    ----------
        - OneValueIndicators: Dictionary
            The dictionary where the results must be stored
        - name: str
            The name of the model (i.e OLS, GLSAR, ...).
        - results: statsmodels result object of the model
            The results object of the model
        - Explanatory: Pandas DataFrame
            The DataFrame with the explanatory series
        - NumDecimal: int
            Number of decimals for the numbers calculated

    Return:
    ----------
        - OneValueIndicators: Dictionary
            The inputed dictionary augmented with new indicators
    """

    if name == "RecursiveLS":
        # Model Parameters
        OneValueIndicators[name + " Params"] = Statsmodels_Params(name, results, Explanatory, NumDecimal)

        # Standard Error of the parameters
        OneValueIndicators[name + " Std. Error Params"] = Statsmodels_Std_Error_Params(name, results, Explanatory,
                                                                                       NumDecimal)

        # P-Values
        OneValueIndicators[name + " P-Values"] = Statsmodels_PValues(name, results, Explanatory, NumDecimal)

        # F-test
        OneValueIndicators[name + " F-Test"] = Statsmodels_FTest(results, Explanatory, NumDecimal)

        # TTest
        OneValueIndicators[name + " T-Test"] = Statsmodels_TTest(results, Explanatory, NumDecimal)

        # Scale
        OneValueIndicators[name + " Scale"] = round(results.scale, NumDecimal)

        # Mean Squared Error of the Residuals
        OneValueIndicators[name + " Mean Squared Error of the Residuals"] = round(results.mse_resid, NumDecimal)

        # Mean Squared Error of the Model
        OneValueIndicators[name + " Mean Squared Error of the Model"] = round(results.mse_model, NumDecimal)

        # Total Mean Squared Error
        OneValueIndicators[name + " Total Mean Squared Error"] = round(results.mse_total, NumDecimal)

        # Confidence Interval 95%
        OneValueIndicators[name + " Confidence Interval 95%"] = results.conf_int(1 - 0.95)

        # Confidence Interval 99%
        # OneValueIndicators[name + " Confidence Interval 99%"] = results.conf_int(1-0.99)

        # Number Of Observations
        OneValueIndicators[name + " Nobs"] = round(results.nobs, NumDecimal)

    elif "GLM" in name:

        # Model Parameters
        OneValueIndicators[name + " Params"] = Statsmodels_Params(name, results, Explanatory, NumDecimal)

        # Standard Error of the parameters
        OneValueIndicators[name + " Std. Error Params"] = Statsmodels_Std_Error_Params(name, results, Explanatory,
                                                                                       NumDecimal)

        # P-Values
        OneValueIndicators[name + " P-Values"] = Statsmodels_PValues(name, results, Explanatory, NumDecimal)

        # F-test
        OneValueIndicators[name + " F-Test"] = Statsmodels_FTest(results, Explanatory, NumDecimal)

        # TTest
        OneValueIndicators[name + " T-Test"] = Statsmodels_TTest(results, Explanatory, NumDecimal)

        # Scale
        OneValueIndicators[name + " Scale"] = round(results.scale, NumDecimal)

        # Confidence Interval 95%
        OneValueIndicators[name + " Confidence Interval 95%"] = results.conf_int(1 - 0.95)

        # Confidence Interval 99%
        # OneValueIndicators[name + " Confidence Interval 99%"] = results.conf_int(1-0.99)

        # Number Of Observations
        OneValueIndicators[name + " Nobs"] = round(results.nobs, NumDecimal)

    elif name == "AR":
        # Model Parameters
        OneValueIndicators[name + " Params"] = Statsmodels_Params(name, results, Explanatory, NumDecimal)

        # Standard Error of the parameters
        OneValueIndicators[name + " Std. Error Params"] = Statsmodels_Std_Error_Params(name, results, Explanatory,
                                                                                       NumDecimal)

        # P-Values
        OneValueIndicators[name + " P-Values"] = Statsmodels_PValues(name, results, Explanatory, NumDecimal)

        # F-test
        OneValueIndicators[name + " F-Test"] = Statsmodels_FTest(results, Explanatory, NumDecimal)

        # TTest
        OneValueIndicators[name + " T-Test"] = Statsmodels_TTest(results, Explanatory, NumDecimal)

        # Scale
        OneValueIndicators[name + " Scale"] = round(results.scale, NumDecimal)

        # Confidence Interval 95%
        OneValueIndicators[name + " Confidence Interval 95%"] = results.conf_int(1 - 0.95)

        # Confidence Interval 99%
        # OneValueIndicators[name + " Confidence Interval 99%"] = results.conf_int(1-0.99)

        # Number Of Observations
        OneValueIndicators[name + " Nobs"] = round(results.nobs, NumDecimal)

    elif name == "ARMA":

        # Model Parameters
        OneValueIndicators[name + " Params"] = Statsmodels_Params(name, results, Explanatory, NumDecimal)

        # Standard Error of the parameters
        OneValueIndicators[name + " Std. Error Params"] = Statsmodels_Std_Error_Params(name, results, Explanatory,
                                                                                       NumDecimal)

        # P-Values
        OneValueIndicators[name + " P-Values"] = Statsmodels_PValues(name, results, Explanatory, NumDecimal)

        # F-test
        # OneValueIndicators[name + " F-Test"] = Statsmodels_FTest(results, Explanatory, NumDecimal)

        # TTest
        # OneValueIndicators[name + " T-Test"] = Statsmodels_TTest(results, Explanatory, NumDecimal)

        # Scale
        OneValueIndicators[name + " Scale"] = round(results.scale, NumDecimal)

        # Confidence Interval 95%
        OneValueIndicators[name + " Confidence Interval 95%"] = results.conf_int(1 - 0.95)

        # Confidence Interval 99%
        # OneValueIndicators[name + " Confidence Interval 99%"] = results.conf_int(1-0.99)

        # Number Of Observations
        OneValueIndicators[name + " Nobs"] = round(results.nobs, NumDecimal)

    elif name == "ARIMA":
        # Model Parameters
        OneValueIndicators[name + " Params"] = Statsmodels_Params(name, results, Explanatory, NumDecimal)

        # Standard Error of the parameters
        OneValueIndicators[name + " Std. Error Params"] = Statsmodels_Std_Error_Params(name, results, Explanatory,
                                                                                       NumDecimal)

        # P-Values
        OneValueIndicators[name + " P-Values"] = Statsmodels_PValues(name, results, Explanatory, NumDecimal)

        # Scale
        OneValueIndicators[name + " Scale"] = round(results.scale, NumDecimal)

        # Confidence Interval 95%
        OneValueIndicators[name + " Confidence Interval 95%"] = results.conf_int(1 - 0.95)

        # Confidence Interval 99%
        # OneValueIndicators[name + " Confidence Interval 99%"] = results.conf_int(1-0.99)

        # Number Of Observations
        OneValueIndicators[name + " Nobs"] = round(results.nobs, NumDecimal)


    else:
        # Model Parameters
        OneValueIndicators[name + " Params"] = Statsmodels_Params(name, results, Explanatory, NumDecimal)

        # Standard Error of the parameters
        OneValueIndicators[name + " Std. Error Params"] = Statsmodels_Std_Error_Params(name, results, Explanatory,
                                                                                       NumDecimal)

        # P-Values
        OneValueIndicators[name + " P-Values"] = Statsmodels_PValues(name, results, Explanatory, NumDecimal)

        # F-test
        try:
            OneValueIndicators[name + " F-Test"] = Statsmodels_FTest(results, Explanatory, NumDecimal)
        except:
            OneValueIndicators[name + " F-Test"] = "N/A"

        # TTest
        OneValueIndicators[name + " T-Test"] = Statsmodels_TTest(results, Explanatory, NumDecimal)

        # R²
        OneValueIndicators[name + " R²"] = round(results.rsquared, NumDecimal)

        # R² Adjusted
        OneValueIndicators[name + " R² Adj."] = round(results.rsquared_adj, NumDecimal)

        # Scale
        OneValueIndicators[name + " Scale"] = round(results.scale, NumDecimal)

        # Mean Squared Error of the Residuals
        OneValueIndicators[name + " Mean Squared Error of the Residuals"] = round(results.mse_resid, NumDecimal)

        # Mean Squared Error of the Model
        OneValueIndicators[name + " Mean Squared Error of the Model"] = round(results.mse_model, NumDecimal)

        # Total Mean Squared Error
        OneValueIndicators[name + " Total Mean Squared Error"] = round(results.mse_total, NumDecimal)

        # Confidence Interval 95%
        OneValueIndicators[name + " Confidence Interval 95%"] = results.conf_int(1 - 0.95)

        # Confidence Interval 99%
        # OneValueIndicators[name + " Confidence Interval 99%"] = results.conf_int(1-0.99)

        # Number Of Observations
        OneValueIndicators[name + " Nobs"] = round(results.nobs, NumDecimal)

    return OneValueIndicators
