import statsmodels.api as sm
import pandas as pd


def BackwardElimination(x, y, Threshold):
    """
    This function apply a backard elimination for a linear regression model, based on P Value level.

    Argument:
    ----------
        - x: pandas dataframe
            The dependent variables
        - y: pandas dataframe
            The independent variable
        - Threshold: float
            The threshold is the maximum value that the pvalue of a variable can take. If the pvalue of a variable is higher, then the variable is removed from the model.

    Return:
    ----------
        - x: pandas dataframe
            The x dataframe only containing columns of the variables passing the selection

    """

    NumberOfXVariables = x.shape[1]

    for i in range(0, NumberOfXVariables):

        regressor_OLS = sm.OLS(y, x).fit()
        maxPValue = max(regressor_OLS.pvalues)

        if maxPValue > Threshold:

            for j in range(0, NumberOfXVariables - i):

                if (regressor_OLS.pvalues[j].astype(float) == maxPValue):
                    x = x.drop(x.columns[j], axis=1)

    return x