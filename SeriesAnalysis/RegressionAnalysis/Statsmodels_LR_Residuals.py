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


def Statsmodels_LR_Residuals(df, results, name):
    """
    This function is used to get the Residuals of a statsmodels model results

    Arguments:
    ----------
        - df: Pandas DataFrame
            The DataFrame where the values will be stored (inner merging on index)
        - name: str
            The name of the model (i.e OLS, GLSAR, ...).
        - results: statsmodels result object of the model
            The results object of the model

    Return:
    ----------
        - df: Pandas DataFrame
            The inputed DataFrame augmented of the new series
    """
    if name == "OLS":

        # Residuals
        Residuals = pd.DataFrame(results.resid)
        Residuals.index = df.index
        Residuals.rename(inplace=True, columns={Residuals.columns[0]: name + " Residuals"})
        df = df.merge(Residuals, how="inner", left_index=True, right_index=True)

    elif name == "GLSAR":

        # Residuals
        Residuals = pd.DataFrame(results.resid)
        Residuals.index = df.index  # ??? The length of the two index are different. Supposed that the Residuals start at t=0 and have an extra value at the end
        Residuals.rename(inplace=True, columns={Residuals.columns[0]: name + " Residuals"})
        df = df.merge(Residuals, how="inner", left_index=True, right_index=True)

        # Residuals Pearson (normalized)
        ResidualsPearson = pd.DataFrame(results.resid_pearson)
        ResidualsPearson.index = df.iloc[:-1].index
        ResidualsPearson.rename(inplace=True, columns={ResidualsPearson.columns[0]: name + " Residuals Pearson"})
        df = df.merge(ResidualsPearson, how="inner", left_index=True, right_index=True)

    elif name == "RecursiveLS":

        # Residuals
        Residuals = pd.DataFrame(results.resid)
        Residuals = pd.DataFrame(Residuals.iloc[:-1])  # ??? why a length difference
        Residuals.index = df.index
        Residuals.rename(inplace=True, columns={Residuals.columns[0]: name + " Residuals"})
        df = df.merge(Residuals, how="inner", left_index=True, right_index=True)

        # Residuals Recursive
        ResidualsRecursive = pd.DataFrame(results.resid_recursive)
        ResidualsRecursive = pd.DataFrame(ResidualsRecursive.iloc[:-1])
        ResidualsRecursive.index = df.index
        ResidualsRecursive.rename(inplace=True, columns={ResidualsRecursive.columns[0]: name + " Residuals Recursive"})
        df = df.merge(ResidualsRecursive, how="inner", left_index=True, right_index=True)

    elif name == "QuantReg":

        # Residuals
        Residuals = pd.DataFrame(results.resid)
        Residuals = pd.DataFrame(Residuals.iloc[:-1])
        Residuals.index = df.index
        Residuals.rename(inplace=True, columns={Residuals.columns[0]: name + " Residuals"})
        df = df.merge(Residuals, how="inner", left_index=True, right_index=True)

        # Residuals Pearson (normalized)
        ResidualsPearson = pd.DataFrame(results.resid_pearson)
        ResidualsPearson = pd.DataFrame(ResidualsPearson.iloc[:-1])
        ResidualsPearson.index = df.index
        ResidualsPearson.rename(inplace=True, columns={ResidualsPearson.columns[0]: name + " Residuals Pearson"})
        df = df.merge(ResidualsPearson, how="inner", left_index=True, right_index=True)

    elif "GLM" in name:

        # Residuals WOrking
        ResidualsWorking = pd.DataFrame(results.resid_working)
        ResidualsWorking = pd.DataFrame(ResidualsWorking.iloc[:-1])
        ResidualsWorking.index = df.index
        ResidualsWorking.rename(inplace=True, columns={ResidualsWorking.columns[0]: name + " Residuals Working"})
        df = df.merge(ResidualsWorking, how="inner", left_index=True, right_index=True)

        # Residuals Response
        ResidualsResponse = pd.DataFrame(results.resid_response)
        ResidualsResponse = pd.DataFrame(ResidualsResponse.iloc[:-1])
        ResidualsResponse.index = df.index
        ResidualsResponse.rename(inplace=True, columns={ResidualsResponse.columns[0]: name + " Residuals Response"})
        df = df.merge(ResidualsResponse, how="inner", left_index=True, right_index=True)

        # Residuals Pearson (normalized)
        ResidualsPearson = pd.DataFrame(results.resid_pearson)
        ResidualsPearson = pd.DataFrame(ResidualsPearson.iloc[:-1])
        ResidualsPearson.index = df.index
        ResidualsPearson.rename(inplace=True, columns={ResidualsPearson.columns[0]: name + " Residuals Pearson"})
        df = df.merge(ResidualsPearson, how="inner", left_index=True, right_index=True)

        # Residuals Deviance
        ResidualsDeviance = pd.DataFrame(results.resid_deviance)
        ResidualsDeviance = pd.DataFrame(ResidualsDeviance.iloc[:-1])
        ResidualsDeviance.index = df.index
        ResidualsDeviance.rename(inplace=True, columns={ResidualsDeviance.columns[0]: name + " Residuals Deviance"})
        df = df.merge(ResidualsDeviance, how="inner", left_index=True, right_index=True)

        # Residuals Anscombe Unscaled
        ResidualsAnscombeUnscaled = pd.DataFrame(results.resid_anscombe_unscaled)
        ResidualsAnscombeUnscaled = pd.DataFrame(ResidualsAnscombeUnscaled.iloc[:-1])
        ResidualsAnscombeUnscaled.index = df.index
        ResidualsAnscombeUnscaled.rename(inplace=True, columns={
            ResidualsAnscombeUnscaled.columns[0]: name + " Residuals Anscombe Unscaled"})
        df = df.merge(ResidualsAnscombeUnscaled, how="inner", left_index=True, right_index=True)

        # Residuals Anscombe Scaled
        ResidualsAnscombeScaled = pd.DataFrame(results.resid_anscombe_scaled)
        ResidualsAnscombeScaled = pd.DataFrame(ResidualsAnscombeScaled.iloc[:-1])
        ResidualsAnscombeScaled.index = df.index
        ResidualsAnscombeScaled.rename(inplace=True, columns={
            ResidualsAnscombeScaled.columns[0]: name + " Residuals Anscombe Scaled"})
        df = df.merge(ResidualsAnscombeScaled, how="inner", left_index=True, right_index=True)

    elif name == "AR":

        # Residuals
        Residuals = pd.DataFrame(results.resid)
        Residuals.index = df.index
        Residuals.rename(inplace=True, columns={Residuals.columns[0]: name + " Residuals"})
        df = df.merge(Residuals, how="inner", left_index=True, right_index=True)

    return df
