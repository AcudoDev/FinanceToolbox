import pandas as pd

def Predict_WithCorrectIndex(model, XToPredict):
    """
    This function return the values predicted bu the model with the specific index as a DataFrame
    Useful when merging or comparaing the predicted values with the true values

    Arguments:
    ----------
        - model: model object
            The model object itself, on which the predict method will be applied
        - XToPredict: Pandas DataFrame
            The dataframe to use as the X values

    Return:
    ----------
        - Prediction: Pandas Dataframe
            The predicted values in a Dataframe with the Prediction.index equal to the YIndex

    """

    Prediction = pd.DataFrame(model.predict(XToPredict))
    Prediction.index = XToPredict.index

    return Prediction