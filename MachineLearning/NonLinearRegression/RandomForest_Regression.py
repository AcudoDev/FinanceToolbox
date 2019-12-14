import FinanceToolbox as FT

from sklearn.ensemble import RandomForestRegressor

import pandas as pd


def RandomForest_Regression(x, y, NumberOfTrees):
    """
    This function compute and output results of a Random Forest Regression.

    Arguments:
    ----------
        - X: pandas dataframe
            Dataframe containing dependent variables
        - y: pandas dataframe
            Dataframe containing independent variables

    Return:
    ----------
        - model: random forests fitted object from sklearn svm class
            The fitted model object
        - Results: pandas dataframe
            Statistics about the model such as the score and MSE
    """

    model = RandomForestRegressor(n_estimators = NumberOfTrees)
    model.fit(X=x, y=y)

    Predict_y = model.predict(x)

    Results = FT.MachineLearning.Metrics.ModelEvaluation(y, Predict_y,
                                                         Indicators = ["Explained Variance Score", "Max Error", "Mean Squared Error", "R² Score"])

    return model, Results