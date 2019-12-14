import FinanceToolbox as FT

from sklearn.svm import SVR

import pandas as pd


def SVR_Regression(x, y, Kernel="rbf"):
    """
    This function compute and output results of a Support Vector Regression.
s
    Arguments:
    ----------
        - X: pandas dataframe
            Dataframe containing dependent variables
        - y: pandas dataframe
            Dataframe containing independent variables
        - Kernel: str
            The name of the Kernel to use for the SVR sklearnobject

    Return:
    ----------
        - model: SVR fitted object from sklearn svm class
            The fitted model object
        - Results: pandas dataframe
            Statistics about the model such as the score and MSE
    """

    model = SVR(kernel=Kernel)
    model.fit(X=x, y=y)

    Predict_y = model.predict(x)

    Results = FT.MachineLearning.Metrics.ModelEvaluation(y, Predict_y,
                                                         Indicators = ["Explained Variance Score", "Max Error", "Mean Squared Error", "R² Score"])

    return model, Results