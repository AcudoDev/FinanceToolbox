import pandas as pd
import numpy as np
import threading

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

def AssetAnalysis(df, data_type, Indicators=None, ColumnMapping=None):
    """

    This function calculate a set of indicators based on a stock price dataframe.

    Arguments:
    ----------
        - df: Pandas DataFrame
            Contain tbe stock price data, with the date as index
        - data_type: str
            This define what contains the df variable:
                - "close": index is the date and only the column called Close will be taken into account
                - "ohlc": index is the date and there are 4 columns called Open, High, Low, Close.
                - "ohlcv": index is the date and there are 5 columns called Open, High, Low, Close, Volume.
        - Indicators: list
            A list containing all the indicators to calculate. If None, all indicators are calculated.
        - ColumnMapping: dict
            A dictionary used for column name correspondance. Default is:
                ColumnMapping = {"Open":"Open",
                              "High":"High",
                              "Low":"Low",
                              "Close":"Close",
                              "Volume":"Volume"}
                if the column name in df are different, i.e Close is called Ipsum, then change the item string from Close to Ipsum.
    Return:
    ----------
        - df: Pandas DataFrame
            The original dataframe with all indicators that are themselves times series
        - OneValueIndicators: dictionary
            A dictionary with indicators that are not time series themselves. In other words, indicators with a "unique value".

    """

    def Merging(df, series, name):
        """
        Merge new data to the global df dataframe.
        """
        newdf = pd.DataFrame(series)
        newdf.rename(inplace=True, columns={newdf.columns[0]: name})
        newdf.index = df.index
        df = df.merge(newdf, how="outer", left_index=True, right_index=True)

        return df

    OneValueIndicators = {}

    NumDecimal = 3

    if Indicators == None:
        Indicators = ["Absolute Price Oscillator", "Center of Gravity", "Chande Momentum Oscillator",
                      "Coppock Curve", "Know Sure Thing", "Moving Average Convergence/Divergence",
                      "Momentum", "Percent Price Oscillator", "Rate Of Change", "Relative Strength Index",
                      "Slope", "TRIX", "True Strength Index", "Double Exponential Moving Average",
                      "Exponential Moving Average", "Fibonacci's Weighted Moving Average", "Hull Moving Average",
                      "Kaufman Adaptative Moving Average", "Linear Regression Moving Average", "MidPoint",
                      "Pascals Weighted Moving Average", "wildeR's Moving Average", "Simple Moving Average",
                      "Symetric Weighted Moving Average", "T3", "Triple Exponential Moving Average",
                      "Triangular Moving Average", "Weighted Moving Average", "Zero Lag Moving Average",
                      "Log Return", "Cumulative Log Return", "Percent Return",
                      "Cumulative Percent Return", "Kurtosis", "Mean Absolute Deviation", "Median",
                      "Quantile 10%", "Quantile 20%", "Quantile 30%", "Quantile 40%", "Quantile 50%",
                      "Quantile 60%", "Quantile 70%", "Quantile 80%", "Quantile 90%", "Skew",
                      "Stdev", "Variance", "ZScore", "Differncing Order 1", "Log Transformed", "Power 2 Transformed",
                      "SQRT Transformed", "Differencing 1 Log Transformed", "Differencing 1 Power 2 Transformed",
                      "Differencing 1 SQRT Transformed", "Holt Winter’s Exponential Smoothing", "Aroon Oscillator",
                      "Decreasing", "Detrend Oscillator",
                      "Linear Decay", "Bollinger Bands", "Donchian Channels", "Awesome Oscillator",
                      "Balance Of Power", "Comodity Channel Index", "Fisher Transform", "RVI",
                      "Stochastic Oscillator", "Ultimate Oscillator", "William's Percent R",
                      "HL2", "HLC3", "Ichimoku", "OHLC4", "ADX", "Archer Moving Average Trends",
                      "Qstick", "Vortex", "Acceleration Bands", "Average True Range",
                      "Keltner Channels", "Mass Index", "Normalized Average True Range", "True Range",
                      "Volume Weighted Average Price", "Volume Weighted Moving Average", "Accumulation/Distribution",
                      "Accumulation/Distribution", "Archer On Balance Volume", "Chaikin Money Flow",
                      "Elder's Force Index", "Ease Of Movement", "Money Flow Index", "Negative Volume Index",
                      "On Balance Volume", "Positive Volume Index", "Price Volume", "Price Volume Trend",
                      "Volume Profile", "Kalman Filter", "Savgol Filter", "Linear Filter - Convolution", "Last",
                      "1d Change (%)", "5d Change (%)", "10d Change (%)",
                      "1d Change", "5d Change", "10d Change", "Hurst Exponent", "Augmented Dickey Fuller", "KPSS",
                      "Speed Mean Reversion",
                      "Global Stationarity", "Halflife"]

    DefaultMapping = {"Open": "Open",
                      "High": "High",
                      "Low": "Low",
                      "Close": "Close",
                      "Volume": "Volume"}

    if ColumnMapping != None:
        for key1 in ColumnMapping:
            for key2 in DefaultMapping:
                if key1 == key2:
                    DefaultMapping[key2] = ColumnMapping[key1]

    # Set series
    if data_type == "close":
        close = df[DefaultMapping["Close"]]
    elif data_type == "ohlc":
        open_ = df[DefaultMapping["Open"]]
        high = df[DefaultMapping["High"]]
        low = df[DefaultMapping["Low"]]
        close = df[DefaultMapping["Close"]]
    elif data_type == "ohlcv":
        open_ = df[DefaultMapping["Open"]]
        high = df[DefaultMapping["High"]]
        low = df[DefaultMapping["Low"]]
        close = df[DefaultMapping["Close"]]
        volume = df[DefaultMapping["Volume"]]

    # Close column name
    closename = DefaultMapping["Close"]

    #Storing initial column names
    InitialColumnNames = df.columns

    ##################################################
    ########## PART 1: Time Series Indicators
    ##################################################

    """
    ########## Section 1: Common Indicators for "close" and "ohlc" data
    """

    ### Momentum

    # Absolute Price Oscillator
    if "Absolute Price Oscillator" in Indicators:
        df.ta.apo(close, fast=None, slow=None, offset=None, append=True)

    # Center of Gravity
    if "Center of Gravity" in Indicators:
        df.ta.cg(close, length=None, offset=None, append=True)

    # Chande Momentum Oscillator
    if "Chande Momentum Oscillator" in Indicators:
        df.ta.cmo(close, length=None, drift=None, offset=None, append=True)

    # Coppock Curve
    if "Coppock Curve" in Indicators:
        df.ta.coppock(close, length=None, fast=None, slow=None, offset=None, append=True)

    # Know Sure Thing
    if "Know Sure Thing" in Indicators:
        df.ta.kst(close, roc1=None, roc2=None, roc3=None, roc4=None, sma1=None, sma2=None, sma3=None, sma4=None,
                  signal=None, drift=None, offset=None, append=True)

    # Moving Average Convergence/Divergence
    if "Moving Average Convergence/Divergence" in Indicators:
        df.ta.macd(close, fast=None, slow=None, signal=None, offset=None, append=True)

    # Momentum
    if "Momentum" in Indicators:
        df.ta.mom(close, length=None, offset=None, append=True)

    # Percent Price Oscillator
    if "Percent Price Oscillator" in Indicators:
        df.ta.ppo(close, fast=None, slow=None, signal=None, offset=None, append=True)

    # Rate Of Change
    if "Rate Of Change" in Indicators:
        df.ta.roc(close, length=None, offset=None, append=True)

    # Relative Strength Index
    if "Relative Strength Index" in Indicators:
        df.ta.rsi(close, length=None, drift=None, offset=None, append=True)

    # Slope
    if "Slope" in Indicators:
        df.ta.slope(close, length=None, as_angle=None, to_degrees=None, offset=None, append=True)

    # TRIX
    if "TRIX" in Indicators:
        df.ta.trix(close, length=None, drift=None, offset=None, append=True)

    # True Strength Index
    if "True Strength Index" in Indicators:
        df.ta.tsi(close, fast=None, slow=None, drift=None, offset=None, append=True)

        ### Overlap

    # Double Exponential Moving Average
    if "Double Exponential Moving Average" in Indicators:
        df.ta.dema(close, length=None, offset=None, append=True)

    # Exponential Moving Average
    if "Exponential Moving Average" in Indicators:
        df.ta.ema(close, length=None, offset=None, append=True)

    # Fibonacci's Weighted Moving Average
    if "Fibonacci's Weighted Moving Average" in Indicators:
        df.ta.fwma(close, length=None, asc=None, offset=None, append=True)

    # Hull Moving Average
    if "Hull Moving Average" in Indicators:
        df.ta.hma(close, length=None, offset=None, append=True)

    # Kaufman Adaptative Moving Average
    if "Kaufman Adaptative Moving Average" in Indicators:
        df.ta.kama(close, length=None, fast=None, slow=None, drift=None, offset=None, append=True)

    # Linear Regression Moving Average
    if "Linear Regression Moving Average" in Indicators:
        df.ta.linreg(close, length=None, offset=None, append=True)

    # MidPoint
    if "MidPoint" in Indicators:
        df.ta.midpoint(close, length=None, offset=None, append=True)

    # Pascals Weighted Moving Average
    if "Pascals Weighted Moving Average" in Indicators:
        df.ta.pwma(close, length=None, asc=None, offset=None, append=True)

    # wildeR's Moving Average
    if "wildeR's Moving Average" in Indicators:
        df.ta.rma(close, length=None, offset=None, append=True)

    # Simple Moving Average
    if "Simple Moving Average" in Indicators:
        df.ta.sma(close, length=None, offset=None, append=True)

    # Symetric Weighted Moving Average
    if "Symetric Weighted Moving Average" in Indicators:
        df.ta.swma(close, length=None, asc=None, offset=None, append=True)

    # T3
    if "T3" in Indicators:
        df.ta.t3(close, length=None, a=None, offset=None, append=True)

    # Triple Exponential Moving Average
    if "Triple Exponential Moving Average" in Indicators:
        df.ta.tema(close, length=None, offset=None, append=True)

    # Triangular Moving Average
    if "Triangular Moving Average" in Indicators:
        df.ta.trima(close, length=None, offset=None, append=True)

    # Weighted Moving Average
    if "Weighted Moving Average" in Indicators:
        df.ta.wma(close, length=None, asc=None, offset=None, append=True)

    # Zero Lag Moving Average
    if "Zero Lag Moving Average" in Indicators:
        df.ta.zlma(close, length=None, offset=None, mamode=None, append=True)

        ### Performance

    # Log Return
    if "Log Return" in Indicators:
        df.ta.log_return(close, length=None, cumulative=False, offset=None, append=True)

    # Cumulative Log Return
    if "Cumulative Log Return" in Indicators:
        df.ta.log_return(close, length=None, cumulative=True, offset=None, append=True)

    # Percent Return
    if "Percent Return" in Indicators:
        df.ta.percent_return(close, length=None, cumulative=False, offset=None, append=True)

    # Cumulative Percent Return
    if "Cumulative Percent Return" in Indicators:
        df.ta.percent_return(close, length=None, cumulative=True, offset=None, append=True)

    # Trend Return
    # if "Trend Return" in Indicators:
    # df.ta.trend_return(close, trend, log=None, cumulative=None
    # NotImplemented

    ### Statistics

    # Kurtosis
    if "Kurtosis" in Indicators:
        df.ta.kurtosis(close, length=None, offset=None, append=True)

    # Mean Absolute Deviation
    if "Mean Absolute Deviation" in Indicators:
        df.ta.mad(close, length=None, offset=None, append=True)

    # Median
    if "Median" in Indicators:
        df.ta.median(close, length=None, offset=None, append=True)

    # Quantile 10%
    if "Quantile 10%" in Indicators:
        df.ta.quantile(close, length=None, q=0.1, offset=None, append=True)

    # Quantile 20%
    if "Quantile 20%" in Indicators:
        df.ta.quantile(close, length=None, q=0.2, offset=None, append=True)

    # Quantile 30%
    if "Quantile 30%" in Indicators:
        df.ta.quantile(close, length=None, q=0.3, offset=None, append=True)

    # Quantile 40%
    if "Quantile 40%" in Indicators:
        df.ta.quantile(close, length=None, q=0.4, offset=None, append=True)

    # Quantile 50%
    if "Quantile 50%" in Indicators:
        df.ta.quantile(close, length=None, q=0.5, offset=None, append=True)

    # Quantile 60%
    if "Quantile 60%" in Indicators:
        df.ta.quantile(close, length=None, q=0.6, offset=None, append=True)

    # Quantile 70%
    if "Quantile 70%" in Indicators:
        df.ta.quantile(close, length=None, q=0.7, offset=None, append=True)

    # Quantile 80%
    if "Quantile 80%" in Indicators:
        df.ta.quantile(close, length=None, q=0.8, offset=None, append=True)

    # Quantile 90%
    if "Quantile 90%" in Indicators:
        df.ta.quantile(close, length=None, q=0.9, offset=None, append=True)

    # Skew
    if "Skew" in Indicators:
        df.ta.skew(close, length=None, offset=None, append=True)

    # Stdev
    if "Stdev" in Indicators:
        df.ta.stdev(close, length=None, offset=None, append=True)

    # Variance
    if "Variance" in Indicators:
        df.ta.variance(close, length=None, offset=None, append=True)

    # ZScore
    if "ZScore" in Indicators:
        df.ta.zscore(close, length=None, std=None, offset=None, append=True)

    # Depreciated
    # Geometric ZScore
    # if "Geometric ZScore" in Indicators:
    # GZScore = gzscore(close)
    # Merging(GZScore, "GZScore")

    # Differencing Order 1
    if "Differencing Order 1" in Indicators:
        df["Differencing Order 1"] = close - close.shift(1)

    # Log Transformed
    if "Log Transformed" in Indicators:
        df["Log Transformed"] = np.log(close)

    # Power 2 Transformed
    if "Power 2 Transformed" in Indicators:
        df["Power 2 Transformed"] = np.power(close, 2)

    # SQRT Transformed
    if "SQRT Transformed" in Indicators:
        df["SQRT Transformed"] = np.sqrt(close)

    # Differencing 1 Log Transformed
    if "Differencing 1 Log Transformed" in Indicators:
        df["Differencing 1 Log Transformed"] = np.log(close) - np.log(close).shift(1)

    # Differencing 1 Power 2 Transformed
    if "Differencing 1 Power 2 Transformed" in Indicators:
        df["Differencing 1 Power 2 Transformed"] = np.power(close, 2) - np.power(close, 2).shift(1)

    # Differencing 1 SQRT Transformed
    if "Differencing 1 SQRT Transformed" in Indicators:
        df["Differencing 1 SQRT Transformed"] = np.sqrt(close) - np.sqrt(close).shift(1)

        ### Trend

    # Aroon Oscillator
    if "Aroon Oscillator" in Indicators:
        df.ta.aroon(close, length=None, offset=None, append=True)

    # Decreasing
    if "Decreasing" in Indicators:
        df.ta.decreasing(close, length=None, asint=True, offset=None, append=True)

    # Detrend Oscillator
    if "Detrend Oscillator" in Indicators:
        df.ta.dpo(close, length=None, centered=True, offset=None, append=True)

    # Not working
    # Increasing
    # if "Increasing" in Indicators:
    # df.ta.increasing(close, length=None, asint=True, offset=None)

    # Linear Decay
    if "Linear Decay" in Indicators:
        df.ta.linear_decay(close, length=None, offset=None, append=True)

    # Not working
    # Long Run
    # if "Long Run" in Indicators:
    # fast=5
    # slow=15
    # df.ta.long_run(fast, slow, length=None, offset=None)

    # Not working
    # Short Run
    # if "Short Run" in Indicators:
    # df.ta.short_run(fast, slow, length=None, offset=None)

    ### Volatility

    # Bollinger Bands
    if "Bollinger Bands" in Indicators:
        df.ta.bbands(close, length=None, std=None, mamode=None, offset=None, append=True)

    # Donchian Channels
    if "Donchian Channels" in Indicators:
        df.ta.donchian(close, lower_length=None, upper_length=None, offset=None, append=True)

        ### Volume
        # No Indicator expected

    # Holt Winter’s Exponential Smoothing
    if "Holt Winter’s Exponential Smoothing" in Indicators:
        name = u"Holt Winter's Exponential Smoothing"
        model = statsmodels.tsa.holtwinters.ExponentialSmoothing(close)
        results = model.fit()
        ### One Value Indicators (here to be clearer than splitting the same model into two parts)
        # Not interesting and produce an error : OneValueIndicators[name + " Params"] = Statsmodels_Params(name, results, Explanatory=None, NumDecimal=NumDecimal)
        OneValueIndicators[name + " SSE"] = results.sse
        ### Time Series Indicators
        # Fitted Values
        df = Merging(df, results.fittedvalues, name + " Fitted Values")
        # Residuals
        df = Statsmodels_LR_Residuals(df, results, name)
        # Level
        df[name + " Level"] = pd.Series(results.level)
        # Slope
        df[name + " Slope"] = pd.Series(results.slope)
        # Season
        df[name + " Season"] = pd.Series(results.season)

    """
    ########## Section 2: Indicators for "ohlc" data
    """

    if data_type == "ohlc" or data_type == "ohlcv":

        ### Momentum

        # Awesome Oscillator
        if "Awesome Oscillator" in Indicators:
            df.ta.ao(high, low, fast=None, slow=None, offset=None, append=True)

        # Balance Of Power
        if "Balance Of Power" in Indicators:
            df.ta.bop(open_, high, low, close, offset=None, append=True)

        # Comodity Channel Index
        if "Comodity Channel Index" in Indicators:
            df.ta.cci(high, low, close, length=None, c=None, offset=None, append=True)

        # Fisher Transform
        if "Fisher Transform" in Indicators:
            df.ta.fisher(high, low, length=None, offset=None, append=True)

        # RVI
        if "RVI" in Indicators:
            df.ta.rvi(open_, high, low, close, length=None, swma_length=None, offset=None, append=True)

        # Stochastic Oscillator
        if "Stochastic Oscillator" in Indicators:
            stoch(high, low, close, fast_k=None, slow_k=None, slow_d=None, offset=None, append=True)

        # Ultimate Oscillator
        if "Ultimate Oscillator" in Indicators:
            df.ta.uo(high, low, close, fast=None, medium=None, slow=None, fast_w=None, medium_w=None, slow_w=None,
                     drift=None, offset=None, append=True)

        # William's Percent R
        if "William's Percent R" in Indicators:
            df.ta.willr(high, low, close, length=None, offset=None, append=True)

            ### Overlap

        # HL2
        if "HL2" in Indicators:
            df.ta.hl2(high, low, offset=None, append=True)

        # HLC3
        if "HLC3" in Indicators:
            df.ta.hlc3(high, low, close, offset=None, append=True)

        # Ichimoku
        if "Ichimoku" in Indicators:
            df.ta.ichimoku(high, low, close, tenkan=None, kijun=None, senkou=None, offset=None, append=True)

        # OHLC4
        if "OHLC4" in Indicators:
            df.ta.ohlc4(open_, high, low, close, offset=None, append=True)

            ### Performance

            ### Statistics

            ### Trend

        # ADX
        if "ADX" in Indicators:
            df.ta.adx(high, low, close, length=None, drift=None, offset=None, append=True)

        # Archer Moving Average Trends
        if "Archer Moving Average Trends" in Indicators:
            df.ta.amat(close=None, fast=None, slow=None, mamode=None, lookback=None, offset=None, append=True)

        # Qstick
        if "Qstick" in Indicators:
            df.ta.qstick(open_, close, length=None, offset=None, append=True)

        # Vortex
        if "Vortex" in Indicators:
            df.ta.vortex(high, low, close, length=None, drift=None, offset=None, append=True)

            ### Volatility

        # Acceleration Bands
        if "Acceleration Bands" in Indicators:
            df.ta.accbands(high, low, close, length=None, c=None, drift=None, mamode=None, offset=None, append=True)

        # Average True Range
        if "Average True Range" in Indicators:
            df.ta.atr(high, low, close, length=None, mamode=None, drift=None, offset=None, append=True)

        # Keltner Channels
        if "Keltner Channels" in Indicators:
            df.ta.kc(high, low, close, length=None, scalar=None, mamode=None, offset=None, append=True)

        # Mass Index
        if "Mass Index" in Indicators:
            df.ta.massi(high, low, fast=None, slow=None, offset=None, append=True)

        # Normalized Average True Range
        if "Normalized Average True Range" in Indicators:
            df.ta.natr(high, low, close, length=None, mamode=None, drift=None, offset=None, append=True)

        # True Range
        if "True Range" in Indicators:
            df.ta.true_range(high, low, close, drift=None, offset=None, append=True)

            ### Volume
            # No Indicator Expected

    """
    ########## Section 3: Indicators for "ohlcv" and "ohlc" data
    """

    if data_type == "ohlcv":

        ### Momentum

        ### Overlap

        # Volume Weighted Average Price
        if "Volume Weighted Average Price" in Indicators:
            df.ta.vwap(high, low, close, volume, offset=None, append=True)

        # Volume Weighted Moving Average
        if "Volume Weighted Moving Average" in Indicators:
            df.ta.vwma(close, volume, length=None, offset=None, append=True)

            ### Performance

            ### Statistics

            ### Trend

            ### Volatility

            ### Volume

        # Accumulation/Distribution
        if "Accumulation/Distribution" in Indicators:
            df.ta.ad(high, low, close, volume, open_=None, offset=None, append=True)

        # Accumulation/Distribution
        if "Accumulation/Distribution" in Indicators:
            df.ta.adosc(high, low, close, volume, open_=None, fast=None, slow=None, offset=None, append=True)

        # Archer On Balance Volume
        if "Archer On Balance Volume" in Indicators:
            df.ta.aobv(close, volume, fast=None, slow=None, mamode=None, max_lookback=None, min_lookback=None,
                       offset=None, append=True)

        # Chaikin Money Flow
        if "Chaikin Money Flow" in Indicators:
            df.ta.cmf(high, low, close, volume, open_=None, length=None, offset=None, append=True)

        # Elder's Force Index
        if "Elder's Force Index" in Indicators:
            df.ta.efi(close, volume, length=None, drift=None, mamode=None, offset=None, append=True)

        # Ease Of Movement
        if "Ease Of Movement" in Indicators:
            df.ta.eom(high, low, close, volume, length=None, divisor=None, drift=None, offset=None, append=True)

        # Money Flow Index
        if "Money Flow Index" in Indicators:
            df.ta.mfi(high, low, close, volume, length=None, drift=None, offset=None, append=True)

        # Negative Volume Index
        if "Negative Volume Index" in Indicators:
            df.ta.nvi(close, volume, length=None, initial=None, offset=None, append=True)

        # On Balance Volume
        if "On Balance Volume" in Indicators:
            df.ta.obv(close, volume, offset=None, append=True)

        # Positive Volume Index
        if "Positive Volume Index" in Indicators:
            df.ta.pvi(close, volume, length=None, initial=None, offset=None, append=True)

        # Price Volume
        if "Price Volume" in Indicators:
            df.ta.pvol(close, volume, offset=None, append=True)

        # Price Volume Trend
        if "Price Volume Trend" in Indicators:
            df.ta.pvt(close, volume, drift=None, offset=None, append=True)

        # Volume Profile
        if "Volume Profile" in Indicators:
            df.ta.vp(close, volume, width=None, append=True)

    ##################################################
    ##### PART 2: Equivalent Kalman & Other Filters Series
    ##################################################
    if "Kalman Filter" in Indicators:
        try:
            df = df.merge(FT.SeriesAnalysis.Filters.KF_x1z1(pd.DataFrame(df[col]).dropna(), Q=0.01), how="outer", left_index=True,
                          right_index=True)
        except:
            pass
    if "Savgol Filter" in Indicators:
        try:
            df = df.merge(pd.DataFrame(FT.SeriesAnalysis.Filters.SavgolFilter(df[col], 51, 3)).dropna(), how="outer", left_index=True,
                          right_index=True)
        except:
            pass

        # Hodrick-Prescott filter
        # The smoothing parameter for daily data can range from 1e+4 to 1e+8. Any new data provided can change past data.
        # Seems to be not appropriate for finance.

        # Christiano Fitzgerald asymmetric, random walk filter
        # Noti Implemented - variables settings

        # Baxter-King bandpass filter
        # Not Implemented - variables settings

        # if "Linear Filter - Convolution" in Indicators:
        #    df["Linear Filter - Convolution"] = statsmodels.tsa.filters.filtertools.convolution_filter(close.values, filt=1, nsides=2)
        # Not working

    ##################################################
    ##### PART 3: Unique Value Indicators
    ##################################################

    for col in df.columns:

        col = str(col)

        if "Last" in Indicators:
            OneValueIndicators["Last " + col] = df[col].iloc[-1]

        if "1d Change (%)" in Indicators:
            OneValueIndicators["1d Change (%) " + col] = df[col].iloc[-1] / df[col].iloc[-2] - 1

        if "5d Change (%)" in Indicators:
            OneValueIndicators["5d Change (%) " + col] = df[col].iloc[-1] / df[col].iloc[-6] - 1

        if "10d Change (%)" in Indicators:
            OneValueIndicators["10d Change (%) " + col] = df[col].iloc[-1] / df[col].iloc[-11] - 1

        if "1d Change" in Indicators:
            OneValueIndicators["1d Change " + col] = df[col].iloc[-1] - df[col].iloc[-2]

        if "5d Change" in Indicators:
            OneValueIndicators["5d Change " + col] = df[col].iloc[-1] - df[col].iloc[-6]

        if "10d Change" in Indicators:
            OneValueIndicators["10d Change " + col] = df[col].iloc[-1] - df[col].iloc[-11]

        if "Hurst Exponent" in Indicators:
            try:
                OneValueIndicators["HE Value " + col], OneValueIndicators["HE Interpretation " + col] = FT.SeriesAnalysis.HurstExponent.Hurst_Exponent(
                    df[col].dropna())
            except:
                pass

        if "Augmented Dickey Fuller" in Indicators:
            try:
                OneValueIndicators["ADF " + col] = FT.SeriesAnalysis.Stationarity.AugmentedDickeyFuller(df[col].dropna())
            except:
                pass

        if "KPSS" in Indicators:
            OneValueIndicators["KPSS " + col] = FT.SeriesAnalysis.Stationarity.KPSS(df[col].dropna())

        if "Global Stationarity" in Indicators:
            try:
                OneValueIndicators["Global Stationarity " + col] = FT.SeriesAnalysis.Stationarity.GlobalStationarityInterpretation(
                    FT.SeriesAnalysis.Stationarity.AugmentedDickeyFuller(df[col].dropna()), FT.SeriesAnalysis.Stationarity.KPSS(df[col].dropna()))
            except:
                pass

        if "Speed Mean Reversion" in Indicators:
            try:
                OneValueIndicators["Speed MR " + col] = FT.SeriesAnalysis.SpeedMeanReversion.SpeedMeanReversion(df[col].dropna())
            except:
                pass

        if "Halflife" in Indicators:
            try:
                OneValueIndicators["Halflife " + col] = FT.SeriesAnalysis.Halflife.HalfLife(df[col].dropna())
            except:
                pass

    #rounding the values
    for key in OneValueIndicators:
        try:
            OneValueIndicators[key] = round(OneValueIndicators[key], NumDecimal)
        except:
            pass
    OneValueIndicators = pd.DataFrame.from_dict(OneValueIndicators, orient="index")

    #renaming columns to integrate the Close Column Name provided
    for col in df.columns:
        if col not in InitialColumnNames:
            df = df.rename(columns={col:closename + "_" + col})

    return df, OneValueIndicators