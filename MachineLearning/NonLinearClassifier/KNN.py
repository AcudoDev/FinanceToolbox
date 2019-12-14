import FinanceToolbox as FT

from sklearn.neighbors import KNeighborsClassifier

import pandas as pd


def KNN(x, y):
    """
    This function compute and output results of a KNN Classifier.

    Arguments:
    ----------
        - X: pandas dataframe
            Dataframe containing dependent variables
        - y: pandas dataframe
            Dataframe containing independent variables

    Return:
    ----------
        - model: KNN fitted object from sklearn svm class
            The fitted model object
        - Results: pandas dataframe
            Statistics about the model such as the score and MSE
    """

    model = KNeighborsClassifier(n_neighbors=5, metric = "minkowski", p=2)
    model.fit(X=x, y=y)

    Predict_y = model.predict(x)

    Results = FT.MachineLearning.Metrics.ModelEvaluation(y, Predict_y)

    return model, Results