import pandas as pd
from sklearn.tree import DecisionTreeRegressor

import FinanceToolbox as FT


def DecisionTree_Regression(x, y):
    """
    This function compute and output results of a Decision Tree Regression.

    Note: this is a non linear and non continuous model. The visualization looks like "stairs" (horizontal line followed by a vertical one, the a horizontal, etc)


    Arguments:
    ----------
        - X: pandas dataframe
            Dataframe containing dependent variables
        - y: pandas dataframe
            Dataframe containing independent variables

    Return:
    ----------
        - model: decision tree fitted object from sklearn decision tree regression class
            The fitted model object
        - Results: pandas dataframe
            Statistics about the model such as the score and MSE
    """

    model = DecisionTreeRegressor()
    model.fit(X=x, y=y)

    Predict_y = model.predict(x)

    Results = FT.MachineLearning.Metrics.ModelEvaluation(y, Predict_y,
                                                         Indicators = ["Explained Variance Score", "Max Error", "Mean Squared Error", "R² Score"])

    #Tree Depth
    Results["Tree Depth"] = model.get_depth()

    #Tree Leaves
    Results["Tree Leaves"] = model.get_n_leaves()

    return model, Results