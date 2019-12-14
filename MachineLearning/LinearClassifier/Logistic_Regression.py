import FinanceToolbox as FT

from sklearn.linear_model import LogisticRegression

import pandas as pd


def Logistic_Regression(x, y):
    """
    This function compute and output results of a Logistic Regression.
s
    Arguments:
    ----------
        - X: pandas dataframe
            Dataframe containing dependent variables
        - y: pandas dataframe
            Dataframe containing independent variables

    Return:
    ----------
        - model: logistic regression fitted object from sklearn svm class
            The fitted model object
        - Results: pandas dataframe
            Statistics about the model such as the score and MSE
    """

    model = LogisticRegression()
    model.fit(X=x, y=y)

    Predict_y = model.predict(x)

    Results = FT.MachineLearning.Metrics.ModelEvaluation(y, Predict_y)

    return model, Results