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

from .Statsmodels_Regression_All_OneValueIndicators import *
from .Statsmodels_FittedValues import *
from .Statsmodels_LR_Residuals import *

def RegressionAnalysis(df, Independent, Explanatory, Indicators, prefix=None):
    """
    This function performs regression models, comparaison between series

    Arguments:
    ----------
        - df: Pandas DataFrame
            Contains the data to be analyzed
        - Independent: str
            The name of column in df for the Independent variable data
        - Explanatory: str or list
            The name of the column in df for the Explanatory variable data. In case of a multivariate analysis, needed to pass a list object of all column names.
        - Indicators: list
            The list of the indicators/models names to compute
    Return:
    ----------
        - df: Pandas DataFrame
            - Contains the initial df and all series indicators are added like the Residuals or the Fitted Values
        - OneValueIndicators: Pandas DataFrame
            - Contains all the indicators calculated with only one value like the FTest or the TTest

    """

    if Indicators == None:
        Indicators = ["OLS", "GLSAR", "RecursiveLS", "Yule Walker Order 1", "Yule Walker Order 2",
                      "Yule Walker Order 3", "Burg Order 1", "Burg Order 2", "Burg Order 3",
                      "QuantReg", "GLM Binomial", "GLM Gamma", "GLM Gaussian", "GLM Inverse Gaussian",
                      "GLM Negative Binomial", "GLM Poisson", "GLM Tweedie"
                                                              "AR", "ARMA", "ARIMA", "Granger Causality",
                      "Levinson Durbin", "Cointegration"]

    # Pre-processing
    Independent = df[Independent]
    Independent = pd.DataFrame(Independent)

    Explanatory = df[Explanatory]
    Explanatory = pd.DataFrame(Explanatory)

    y_sm = np.array(Independent).reshape((-1, 1))

    x_sm = np.array(Explanatory)
    x_sm = sm.add_constant(x_sm)

    NumDecimal = 3  # Number of decimals for rounding numbers

    OneValueIndicators = {}

    if prefix == None:
        prefix = ""

    ##################################################
    ##### PART 1: Linear Regression
    ##################################################

    """
    ########## Section 1: OLS
    """
    name = "OLS"

    if name in Indicators:
        name = prefix + name

        model = sm.OLS(y_sm, x_sm)
        results = model.fit()

        ### One Value Indicators

        OneValueIndicators = Statsmodels_Regression_All_OneValueIndicators(OneValueIndicators, name, results,
                                                                           Explanatory, NumDecimal)

        ### Time Series Indicators

        # Fitted Values
        df = Statsmodels_FittedValues(df, results, name)

        # Residuals
        df = Statsmodels_LR_Residuals(df, results, name)

    """
    ########## Section 2: WLS
    """

    ### Not Implemented

    """
    ########## Section 3: GLS
    """

    ### Not Implemented

    """
    ########## Section 4: GLSAR
    """

    name = "GLSAR"

    if name in Indicators:
        name = prefix + name

        model = sm.GLSAR(y_sm, x_sm, 1)
        results = model.iterative_fit(1)

        ### One Value Indicators

        OneValueIndicators = Statsmodels_Regression_All_OneValueIndicators(OneValueIndicators, name, results,
                                                                           Explanatory, NumDecimal)

        ### Time Series Indicators

        # Fitted Values
        df = Statsmodels_FittedValues(df, results, name)

        # Residuals
        df = Statsmodels_LR_Residuals(df, results, name)

    """
    ########## Section 5: RLS
    """

    name = "RecursiveLS"

    if name in Indicators:
        name = prefix + name

        model = sm.RecursiveLS(y_sm, x_sm)
        results = model.fit()

        ### One Value Indicators

        OneValueIndicators = Statsmodels_Regression_All_OneValueIndicators(OneValueIndicators, name, results,
                                                                           Explanatory, NumDecimal)

        OneValueIndicators[name + " Z Value"] = results.zvalues

        ### Time Series Indicators

        # Fitted Values
        df = Statsmodels_FittedValues(df, results, name)

        # Residuals
        df = Statsmodels_LR_Residuals(df, results, name)

        # Cumsum
        # Not Implemented

    """
    ########## Section 6: Yule Walker ORder 1
    """
    name = "Yule Walker Order 1"

    if name in Indicators and len(Explanatory.columns) == 1:
        name = prefix + name

        rho, sigma = statsmodels.regression.linear_model.yule_walker(x_sm[:, 1].flatten(), order=1)

        ### One Value Indicators

        # Rho
        OneValueIndicators[name + " Rho"] = round(rho[0], NumDecimal)

        # Sigma
        OneValueIndicators[name + " Sigma"] = round(sigma, NumDecimal)

    """
    ########## Section 7: Yule Walker ORder 2
    """
    name = "Yule Walker Order 2"

    if name in Indicators and len(Explanatory.columns) == 1:
        name = prefix + name

        rho, sigma = statsmodels.regression.linear_model.yule_walker(x_sm[:, 1].flatten(), order=2)

        ### One Value Indicators

        # Rho
        OneValueIndicators[name + " Rho"] = round(rho[0], NumDecimal)

        # Sigma2
        OneValueIndicators[name + " Sigma"] = round(sigma, NumDecimal)

    """
    ########## Section 8: Yule Walker ORder 3
    """
    name = "Yule Walker Order 3"

    if name in Indicators and len(Explanatory.columns) == 1:
        name = prefix + name

        rho, sigma = statsmodels.regression.linear_model.yule_walker(x_sm[:, 1].flatten(), order=3)

        ### One Value Indicators

        # Rho
        OneValueIndicators[name + " Rho"] = round(rho[0], NumDecimal)

        # Sigma
        OneValueIndicators[name + " Sigma"] = round(sigma, NumDecimal)

    """
    ########## Section 9: Burg's AR(p) ORder 1
    """

    name = "Burg Order 1"

    if name in Indicators and len(Explanatory.columns) == 1:
        name = prefix + name

        rho, sigma2 = statsmodels.regression.linear_model.burg(x_sm[:, 1].flatten(), order=1)

        ### One Value Indicators

        # Rho
        OneValueIndicators[name + " Rho"] = round(rho[0], NumDecimal)

        # Sigma2
        OneValueIndicators[name + " Sigma2"] = round(sigma2, NumDecimal)

    """
    ########## Section 10: Burg's AR(p) ORder 2
    """

    name = "Burg Order 2"

    if name in Indicators and len(Explanatory.columns) == 1:
        name = prefix + name

        rho, sigma2 = statsmodels.regression.linear_model.burg(x_sm[:, 1].flatten(), order=2)

        ### One Value Indicators

        # Rho
        OneValueIndicators[name + " Rho"] = round(rho[0], NumDecimal)

        # Sigma2
        OneValueIndicators[name + " Sigma2"] = round(sigma2, NumDecimal)

    """
    ########## Section 11: Burg's AR(p) ORder 3
    """

    name = "Burg Order 3"

    if name in Indicators and len(Explanatory.columns) == 1:
        name = prefix + name

        rho, sigma2 = statsmodels.regression.linear_model.burg(x_sm[:, 1].flatten(), order=3)

        ### One Value Indicators

        # Rho
        OneValueIndicators[name + " Rho"] = round(rho[0], NumDecimal)

        # Sigma2
        OneValueIndicators[name + " Sigma2"] = round(sigma2, NumDecimal)

    """
    ########## Section 12: Quantile Regression
    """

    name = "QuantReg"

    if name in Indicators:
        name = prefix + name

        model = sm.QuantReg(y_sm, x_sm)
        results = model.fit()

        ### One Value Indicators

        OneValueIndicators = Statsmodels_Regression_All_OneValueIndicators(OneValueIndicators, name, results,
                                                                           Explanatory, NumDecimal)

        ### Time Series Indicators

        # Fitted Values
        df = Statsmodels_FittedValues(df, results, name)

        # Residuals
        df = Statsmodels_LR_Residuals(df, results, name)

    ##################################################
    ##### PART 2: Generalized Linear Models
    ##################################################

    """
    ########## Section 1: GLM Binomial
    """

    name = "GLM Binomial"

    if name in Indicators:
        name = prefix + name

        model = sm.GLM(y_sm, x_sm, family=sm.families.Binomial())
        results = model.fit()

        ### One Value Indicators

        OneValueIndicators = Statsmodels_Regression_All_OneValueIndicators(OneValueIndicators, name, results,
                                                                           Explanatory, NumDecimal)

        OneValueIndicators["Pearson chi2"] = round(results.pearson_chi2, NumDecimal)

        ### Time Series Indicators

        # Fitted Values
        df = Statsmodels_FittedValues(df, results, name)

        # Residuals
        df = Statsmodels_LR_Residuals(df, results, name)

    """
    ########## Section 2: GLM Gamma
    """

    name = "GLM Gamma"

    if name in Indicators:
        name = prefix + name

        model = sm.GLM(y_sm, x_sm, family=sm.families.Gamma())
        results = model.fit()

        ### One Value Indicators

        OneValueIndicators = Statsmodels_Regression_All_OneValueIndicators(OneValueIndicators, name, results,
                                                                           Explanatory, NumDecimal)

        OneValueIndicators["Pearson chi2"] = round(results.pearson_chi2, NumDecimal)

        ### Time Series Indicators

        # Fitted Values
        df = Statsmodels_FittedValues(df, results, name)

        # Residuals
        df = Statsmodels_LR_Residuals(df, results, name)

    """
    ########## Section 3: GLM Gaussian
    """

    name = "GLM Gaussian"

    if name in Indicators:
        name = prefix + name

        model = sm.GLM(y_sm, x_sm, family=sm.families.Gaussian())
        results = model.fit()

        ### One Value Indicators

        OneValueIndicators = Statsmodels_Regression_All_OneValueIndicators(OneValueIndicators, name, results,
                                                                           Explanatory, NumDecimal)

        OneValueIndicators["Pearson chi2"] = round(results.pearson_chi2, NumDecimal)

        ### Time Series Indicators

        # Fitted Values
        df = Statsmodels_FittedValues(df, results, name)

        # Residuals
        df = Statsmodels_LR_Residuals(df, results, name)

    """
    ########## Section 3: GLM InverseGaussian
    """

    name = "GLM Inverse Gaussian"

    if name in Indicators:
        name = prefix + name

        model = sm.GLM(y_sm, x_sm, family=sm.families.InverseGaussian())
        results = model.fit()

        ### One Value Indicators

        OneValueIndicators = Statsmodels_Regression_All_OneValueIndicators(OneValueIndicators, name, results,
                                                                           Explanatory, NumDecimal)

        OneValueIndicators["Pearson chi2"] = round(results.pearson_chi2, NumDecimal)

        ### Time Series Indicators

        # Fitted Values
        df = Statsmodels_FittedValues(df, results, name)

        # Residuals
        df = Statsmodels_LR_Residuals(df, results, name)

    """
    ########## Section 4: GLM NegativeBinomial
    """

    name = "GLM Negative Binomial"

    if name in Indicators:
        name = prefix + name

        model = sm.GLM(y_sm, x_sm, family=sm.families.NegativeBinomial())
        results = model.fit()

        ### One Value Indicators

        OneValueIndicators = Statsmodels_Regression_All_OneValueIndicators(OneValueIndicators, name, results,
                                                                           Explanatory, NumDecimal)

        OneValueIndicators["Pearson chi2"] = round(results.pearson_chi2, NumDecimal)

        ### Time Series Indicators

        # Fitted Values
        df = Statsmodels_FittedValues(df, results, name)

        # Residuals
        df = Statsmodels_LR_Residuals(df, results, name)

    """
    ########## Section 5: GLM Poisson
    """

    name = "GLM Poisson"

    if name in Indicators:
        name = prefix + name

        model = sm.GLM(y_sm, x_sm, family=sm.families.Poisson())
        results = model.fit()

        ### One Value Indicators

        OneValueIndicators = Statsmodels_Regression_All_OneValueIndicators(OneValueIndicators, name, results,
                                                                           Explanatory, NumDecimal)

        OneValueIndicators["Pearson chi2"] = round(results.pearson_chi2, NumDecimal)

        ### Time Series Indicators

        # Fitted Values
        df = Statsmodels_FittedValues(df, results, name)

        # Residuals
        df = Statsmodels_LR_Residuals(df, results, name)

    """
    ########## Section 6: GLM Tweedie
    """

    name = "GLM Tweedie"

    if name in Indicators:
        name = prefix + name

        model = sm.GLM(y_sm, x_sm, family=sm.families.Tweedie())
        results = model.fit()

        ### One Value Indicators

        OneValueIndicators = Statsmodels_Regression_All_OneValueIndicators(OneValueIndicators, name, results,
                                                                           Explanatory, NumDecimal)

        OneValueIndicators["Pearson chi2"] = round(results.pearson_chi2, NumDecimal)

        ### Time Series Indicators

        # Fitted Values
        df = Statsmodels_FittedValues(df, results, name)

        # Residuals
        df = Statsmodels_LR_Residuals(df, results, name)

    ##################################################
    ##### PART 3: Robust Linear Models
    ##################################################

    ##################################################
    ##### PART 4: AR models
    ##################################################

    name = "AR"

    if name in Indicators:
        name = prefix + name

        model = statsmodels.tsa.ar_model.AR(Independent)
        results = model.fit()

        ### One Value Indicators

        OneValueIndicators = Statsmodels_Regression_All_OneValueIndicators(OneValueIndicators, name, results,
                                                                           Explanatory, NumDecimal)

        OneValueIndicators[name + " Final Prediction Error"] = results.fpe

        OneValueIndicators[name + " Hannan-Quinn Information Criterion"] = results.hqic

        OneValueIndicators[name + " Roots"] = results.roots

        ### Time Series Indicators

        # Fitted Values
        df = Statsmodels_FittedValues(df, results, name)

        # Residuals
        df = Statsmodels_LR_Residuals(df, results, name)

    ##################################################
    ##### PART 5: ARMA
    ##################################################

    name = "ARMA"

    if name in Indicators:

        name = prefix + name

        model = statsmodels.tsa.arima_model.ARMA(y_sm, (5, 5), x_sm)
        results = model.fit()

        ### One Value Indicators

        OneValueIndicators = Statsmodels_Regression_All_OneValueIndicators(OneValueIndicators, name, results,
                                                                           Explanatory, NumDecimal)

        OneValueIndicators[name + " AR Params"] = results.arparams

        OneValueIndicators[name + " AR Roots"] = results.arroots

        OneValueIndicators[name + " AR Freq"] = results.arfreq

        OneValueIndicators[name + " Hannan-Quinn Information Criterion"] = results.hqic

        OneValueIndicators[name + " MA Params"] = results.maparams

        try:
            OneValueIndicators[name + " MA Roots"] = results.maroots
        except:
            pass

        try:
            OneValueIndicators[name + " MA Freq"] = results.mafreq
        except:
            pass

        OneValueIndicators[name + " Sigma2"] = results.sigma2

        ### Time Series Indicators

        # Fitted Values
        df = Statsmodels_FittedValues(df, results, name)

        # Residuals
        df = Statsmodels_LR_Residuals(df, results, name)

    ##################################################
    ##### PART 6: ARIMA
    ##################################################

    name = "ARIMA"

    if name in Indicators:

        name = prefix + name

        model = statsmodels.tsa.arima_model.ARIMA(Independent, (2, 2, 2), Explanatory)
        results = model.fit()

        ### One Value Indicators

        OneValueIndicators = Statsmodels_Regression_All_OneValueIndicators(OneValueIndicators, name, results,
                                                                           Explanatory, NumDecimal)

        OneValueIndicators[name + " AR Params"] = results.arparams

        OneValueIndicators[name + " AR Roots"] = results.arroots

        OneValueIndicators[name + " AR Freq"] = results.arfreq

        OneValueIndicators[name + " Hannan-Quinn Information Criterion"] = results.hqic

        OneValueIndicators[name + " MA Params"] = results.maparams

        OneValueIndicators[name + " MA Roots"] = results.maroots

        OneValueIndicators[name + " MA Freq"] = results.mafreq

        OneValueIndicators[name + " Sigma2"] = results.sigma2

        ### Time Series Indicators

        # Fitted Values
        df = Statsmodels_FittedValues(df, results, name)

        # Residuals
        df = Statsmodels_LR_Residuals(df, results, name)

        ##################################################
        ##### PART 7: Univariate Analysis
        ##################################################

        # Granger Causality
        name = "Granger Causality"
        name = prefix + name
        if name in Indicators:
            OneValueIndicators[name] = ts.grangercausalitytests(
                Independent.merge(Explanatory, how="inner", left_index=True, right_index=True), maxlag=10)

        # Levinson Durbin
        name = "Levinson Durbin"
        name = prefix + name
        if name in Indicators:
            OneValueIndicators[name] = ts.levinson_durbin(Independent)

        # Cointegration
        name = "Cointegration"
        name = prefix + name
        if name in Indicators:
            OneValueIndicators[name] = ts.coint(Independent, Explanatory, trend="ct", return_results=False)

    ##################################################
    ##### Not Implemented
    ##################################################

    # BDS Statistic (residuals analysis)
    # Not Implemented

    # Return’s Ljung-Box Q Statistic (AR)
    # Not Implemented
    OneValueIndicators = pd.DataFrame.from_dict(OneValueIndicators, orient="index")

    return df, OneValueIndicators
