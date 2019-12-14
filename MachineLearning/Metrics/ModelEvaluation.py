import pandas as pd
from sklearn.metrics import *

def ModelEvaluation(y_true, y_pred, Indicators=None):
    """
    This function compute indicators to evaluate the model performance.

    Arguments:
    ----------
        - Indicators: list
            The list of indicators name to compute

    Return:
    ----------
        - Results: pandas dataframe
            Dataframe containing the values of calculated indicators

    """

    if Indicators == None:
        Indicators = ["Balanced Accuracy Score", "Confusion Matrix", "F1 Score", "F-Beta Score", "Hamming Loss", "Jaccard Similarity Coefficient Score",
                        "log_loss", "Matthews Correlation Coefficient", "Precision Score", "Recall Score", "Zero-One Loss",
                        "Explained Variance Score", "Max Error", "Mean Squared Error", "R² Score"]

    Results = {}

    """
    ##### Classification
    """
    #Balanced Accuracy Score
    #Percentage of correct predictions
    if "Balanced Accuracy Score" in Indicators:
        BalancedAccuracyScore=balanced_accuracy_score(y_true=y_true, y_pred=y_pred)
        Results["Balanced Accuracy Score"] = BalancedAccuracyScore

    #Confusion Matrix
    if "Confusion Matrix" in Indicators:
        ConfusionMatrix = confusion_matrix(y_true=y_true, y_pred=y_pred)
        Results["Confusion Matrix"] = ConfusionMatrix


    #F1 Score
    #Measure of the accuracy of the prediction
    if "F1 Score" in Indicators:
        F1Score = f1_score(y_true=y_true, y_pred=y_pred)
        Results["F1 Score"] = F1Score

    #F-Beta Score
    if "F-Beta Score" in Indicators:
        FBetaScore = fbeta_score(y_true=y_true, y_pred=y_pred, beta=0.5)
        Results["F-Beta Score"] = FBetaScore

        #Hamming (Average) Loss
    #Percentage of wrong predictions
    if "Hamming Loss" in Indicators:
        HammingLoss = hamming_loss(y_true=y_true, y_pred=y_pred)
        Results["Hamming Loss"] = HammingLoss

    #Jaccard Similarity Coefficient Score
    #Measure of similarity between two datasets - from 0 to 1
    if "Jaccard Similarity Coefficient Score" in Indicators:
        JaccardSimilarityCoefficientScore = jaccard_score(y_true=y_true, y_pred=y_pred)
        Results["Jaccard Similarity Coefficient Score"] = JaccardSimilarityCoefficientScore

        #Log Loss
    #Log loss function - measure of incorrect predictions
    if "Log Loss" in Indicators:
        LogLoss = log_loss(y_true=y_true, y_pred=y_pred)
        Results["Log Loss"] = LogLoss

    #Matthews Correlation Coefficient
    #Measure the "agreement" between predictions and real data points
    #Interval: -1 to +1
    #-1: entire disagreement
    #0: predictions is not better than a random set
    #1: total agreement
    if "Matthews Correlation Coefficient" in Indicators:
        MatthewsCorrelationCoefficient = matthews_corrcoef(y_true=y_true, y_pred=y_pred)
        Results["Matthews Correlation Coefficient"] = MatthewsCorrelationCoefficient

    #Precision Score
    #The precision is intuitively the ability of the classifier not to label as positive a sample that is negative.
    if "Precision Score" in Indicators:
        PrecisionScore = precision_score(y_true=y_true, y_pred=y_pred)
        Results["Precision Score"] = PrecisionScore

    #Recall Score
    #y_true=y_true, y_pred=y_pred
    if "Recall Score" in Indicators:
        RecallScore = recall_score(y_true=y_true, y_pred=y_pred)
        Results["Recall Score"] = RecallScore

    #Zero-One Loss
    if "Zero-One Loss" in Indicators:
        ZeroOneLoss = zero_one_loss(y_true=y_true, y_pred=y_pred)
        Results["Zero-One Loss"] = ZeroOneLoss

    """
    ##### Regression
    """
    #Explained Variance Score
    if "Explained Variance Score" in Indicators:
        ExplainedVarianceScore = explained_variance_score(y_true=y_true, y_pred=y_pred)
        Results["Explained Variance Score"] = ExplainedVarianceScore

    #Max Error
    if "Max Error" in Indicators:
        MaxError = max_error(y_true=y_true, y_pred=y_pred)
        Results["Max Error"] = MaxError

    #Mean Squared Error
    if "Mean Squared Error" in Indicators:
        MeanSquaredError = mean_squared_error(y_true=y_true, y_pred=y_pred)
        Results["Mean Squared Error"] = MeanSquaredError

    #R²
    if "R² Score" in Indicators:
        R2Score = r2_score(y_true=y_true, y_pred=y_pred)
        Results["R² Score"] = R2Score



    return Results