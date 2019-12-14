import pandas as pd
import FinanceToolbox as FT
import statsmodels.api as sm

def RegressionModelComparaison(Df, IndependentVariable, ColToDrop=None):
    """
    This function proceed to the comparaison of different regression models.
    - Simple All In
    - Backward Elmination on an All In model
    - Polynomial Regression with Backward Eliminitation. Degrees: 1, 2
    - SVR Regression
    - Decision Tree
    - Random Forests Regression

    Arguments:
    ----------
        - Df: Pandas Dataframe
            The dataframe containing both independent and explanotory variable. Attention: the dataframe mustn't contain anytthing else than X and y values.
        - IndependentVariable: string
            The name of the column containing the Independent variable
        - ColToDrop: string or list
            The name of the columns to drop

    Return:
    ----------
        - Df_X_Train, Df_X_Test, Df_y_Train, Df_y_Test: Pandas DataFrame
            The corresponding x and y, Train and Test datasets
        - Models: dictionary
            A dictionary where each key is a model and the corresponding element is a list of relevant items for the model.
            For instance, the list of the Simple All In model contains the OLS Series and OneValueIndicators data
        - AllModelComparaison_Prediction_InSample: Pandas DataFrame
            The in sample prediction comparaison of the all models
        - AllModelComparaison_Residual_InSample: Pandas DataFrame
            The in sample residuals for all models


    """

    Models = {}

    # We eliminate the "Combination" column as there are one line per combination name - this can't be a factor
    Df = Df.drop(ColToDrop, axis=1)

    # We apply the following preprocessing tasks:
    # - Split the dataframe into x and y dataframes
    # - Encode Categorical Variables
    # - Normalize results to avoid scale issues
    # - Split x and y into train and test subsets
    # - Add a constant
    # - Sort Index
    Df_X_Train, Df_X_Test, Df_y_Train, Df_y_Test = FT.MachineLearning.Preprocessing.GlobalPreprocessing(
        df=Df,
        Split_x_y=True, IndependentColumnName=[IndependentVariable],
        EncodeCategoricalVariable=True,
        Normalize=True,
        Split_Train_Test=True, TestSetSize=0.2, Randomize=False,
        AddConst=True,
        SortIndex=False)

    # All In Simple Regression results in sample
    Train_SimpleLReg_AllIn_ResultsInSample_Series, Train_SimpleLReg_AllIn_ResultsInSample_OneValueIndicators = FT.SeriesAnalysis.RegressionAnalysis.RegressionAnalysis(
        df=pd.DataFrame(Df_X_Train.iloc[:, 1:]).merge(Df_y_Train, left_index=True, right_index=True),
        Explanatory=Df_X_Train.columns[1:],
        Independent=Df_y_Train.columns,
        Indicators=["OLS"])

    Models["SimpleReg AllIn"] = [Train_SimpleLReg_AllIn_ResultsInSample_Series, Train_SimpleLReg_AllIn_ResultsInSample_OneValueIndicators]

    print("All In Simple Regression Stats")
    print(Train_SimpleLReg_AllIn_ResultsInSample_OneValueIndicators)
    # Prediction in sample VS Observed Graph
    # Train_SimpleLReg_AllIn_ResultsInSample_Series.plot(x="Speed Mean Reversion", y=[IndependentVariable, "OLS Fitted Values"], style="o")

    #Prediction Out of Sample
    model = sm.OLS(Df_y_Train, Df_X_Train)
    model = model.fit()
    SimpleLReg_AllIn_PredOutSample_y = FT.MachineLearning.Metrics.Predict_WithCorrectIndex(model, Df_X_Test)

    # The threshold for elimination based on pvalues
    SL = 0.05

    # Simple Linear Regression Backward Elimination
    # We get the dataframe ONLY with the variables relevant (regarding the elimination step)
    Df_X_Train_SimpleLReg_BackwardElimination = FT.MachineLearning.LinearRegression.BackwardElimination(Df_X_Train,
                                                                                                        Df_y_Train, SL)

    # Get the OLS regression results for in sample
    Train_SimpleLReg_BackwardElimination_ResultsInSample_Series, Train_SimpleLReg_BackwardElimination_ResultsInSample_OneValueIndicators = FT.SeriesAnalysis.RegressionAnalysis.RegressionAnalysis(
        df=Df_X_Train_SimpleLReg_BackwardElimination.merge(Df_y_Train, left_index=True, right_index=True),
        Explanatory=Df_X_Train_SimpleLReg_BackwardElimination.columns,
        Independent=Df_y_Train.columns,
        Indicators=["OLS"])

    Models["SimpleReg BackwardElimination"] = [Train_SimpleLReg_BackwardElimination_ResultsInSample_Series, Train_SimpleLReg_BackwardElimination_ResultsInSample_OneValueIndicators]
    print("Simple Reg Backward Elimination")
    print(Train_SimpleLReg_BackwardElimination_ResultsInSample_OneValueIndicators)
    # Prediction in sample VS Observed Graph
    # Train_SimpleLReg_BackwardElimination_ResultsInSample_Series.plot(x="Speed Mean Reversion", y=[IndependentVariable, "OLS Fitted Values"], style="o")

    #Prediction Out of Sample
    #We first need to reduce Df_X_Test to the variable contained in Df_X_Train_SimpleLReg_BackwardElimination
    Df_X_Test_SimpleLReg_BackwardElimination = Df_X_Test[Df_X_Train_SimpleLReg_BackwardElimination.columns]

    model = sm.OLS(Df_y_Train, Df_X_Train_SimpleLReg_BackwardElimination)
    model = model.fit()
    SimpleLReg_BackwardElimination_PredOutSample_y = FT.MachineLearning.Metrics.Predict_WithCorrectIndex(model, Df_X_Test_SimpleLReg_BackwardElimination)



    # Polynomial Linear Regression Backward Elimination
    # Convert to a polunomial form
    degrees = [1, 2]
    Df_X_Train_PolyLReg = FT.MachineLearning.LinearRegression.ConvertToPolynomial(pd.DataFrame(Df_X_Train.iloc[:, 1:]),
                                                                                  degrees)

    # We get the dataframe ONLY with the variables relevant (regarding the elimination step)
    Df_X_Train_PolyLReg_BackwardElimination = FT.MachineLearning.LinearRegression.BackwardElimination(
        Df_X_Train_PolyLReg, Df_y_Train, SL)

    # Get the OLS regression results
    Train_PolyLReg_BackwardElimination_ResultsInSample_Series, Train_PolyLReg_BackwardElimination_ResultsInSample_OneValueIndicators = FT.SeriesAnalysis.RegressionAnalysis.RegressionAnalysis(
        df=Df_X_Train_PolyLReg_BackwardElimination.merge(Df_y_Train, left_index=True, right_index=True),
        Explanatory=Df_X_Train_PolyLReg_BackwardElimination.columns,
        Independent=Df_y_Train.columns,
        Indicators=["OLS"])

    Models["PolyReg BackwardElimination"] = [Train_PolyLReg_BackwardElimination_ResultsInSample_Series, Train_PolyLReg_BackwardElimination_ResultsInSample_OneValueIndicators]
    print("Poly Reg Backward Elimination Stats")
    print(Train_PolyLReg_BackwardElimination_ResultsInSample_OneValueIndicators)
    # Prediction in sample VS Observed Graph
    # Train_PolyLReg_BackwardElimination_ResultsInSample_Series.plot(x="Speed Mean Reversion", y=[IndependentVariable, "OLS Fitted Values"], style="o")

    # Prediction Out of Sample
    # We first need to reduce Df_X_Test to the variable contained in Df_X_Train_SimpleLReg_BackwardElimination
    Df_X_Test_PolyLReg = FT.MachineLearning.LinearRegression.ConvertToPolynomial(pd.DataFrame(Df_X_Test.iloc[:, 1:]),degrees)
    Df_X_Test_PolyLReg_BackwardElimination = Df_X_Test_PolyLReg[Df_X_Train_PolyLReg_BackwardElimination.columns]

    model = sm.OLS(Df_y_Train, Df_X_Train_PolyLReg_BackwardElimination)
    model = model.fit()
    PolyLReg_BackwardElimination_PredOutSample_y = FT.MachineLearning.Metrics.Predict_WithCorrectIndex(model, Df_X_Test_PolyLReg_BackwardElimination)

    # Support Vector Regression
    SVRModel, SVRResults = FT.MachineLearning.NonLinearRegression.SVR_Regression(Df_X_Train, Df_y_Train)

    Models["SVR"] = [SVRModel, SVRResults]
    print("SVR Stats")
    print(FT.MachineLearning.Metrics.ModelEvaluation(Df_y_Test, SVRModel.predict(Df_X_Test),
                                                     Indicators=["Explained Variance Score", "Max Error",
                                                                 "Mean Squared Error", "R² Score"]))
    #Prediction in sample
    SVR_PredInSample_y = FT.MachineLearning.Metrics.Predict_WithCorrectIndex(SVRModel, Df_X_Train)
    # Prediction in sample VS Observed Graph
    #SVRComp = pd.DataFrame(SVR_PredInSample_y).merge(Df_y_Train, how="inner", left_index=True, right_index=True)
    #SVRComp.plot(style="o")
    # Prediction out sample
    SVR_PredOutSample_y = FT.MachineLearning.Metrics.Predict_WithCorrectIndex(SVRModel, Df_X_Test)

    # Decision Tree Regression
    DecisionTreeModel, DecisionTreeResults = FT.MachineLearning.NonLinearRegression.DecisionTree_Regression(Df_X_Train,
                                                                                                            Df_y_Train)

    Models["DecisionTree"] = [DecisionTreeModel, DecisionTreeResults]
    print("Decision Tree Stats")
    print(FT.MachineLearning.Metrics.ModelEvaluation(Df_y_Test, DecisionTreeModel.predict(Df_X_Test),
                                                     Indicators=["Explained Variance Score", "Max Error",
                                                                 "Mean Squared Error", "R² Score"]))


    # Prediction in sample
    DecisionTree_PredInSample_y = FT.MachineLearning.Metrics.Predict_WithCorrectIndex(DecisionTreeModel, Df_X_Train)
    # Prediction in sample VS Observed
    #DecisionTreeComp = pd.DataFrame(DecisionTree_PredInSample_y).merge(Df_y_Train, how="inner", left_index=True, right_index=True)
    #DecisionTreeComp.plot(style="o")
    # Prediction out sample
    DecisionTree_PredOutSample_y = FT.MachineLearning.Metrics.Predict_WithCorrectIndex(DecisionTreeModel, Df_X_Test)

    # Random Forests
    RandomForestModel, RandomForestResults = FT.MachineLearning.NonLinearRegression.RandomForest_Regression(Df_X_Train,
                                                                                                            Df_y_Train,
                                                                                                            NumberOfTrees=1000)
    Models["RandomForests"] = [RandomForestModel, RandomForestResults, RandomForestModel.feature_importances_]
    print("Random Forests Stats")
    print(FT.MachineLearning.Metrics.ModelEvaluation(Df_y_Test, RandomForestModel.predict(Df_X_Test),
                                                     Indicators=["Explained Variance Score", "Max Error",
                                                                 "Mean Squared Error", "R² Score"]))
    print('Features Importances')
    print(RandomForestModel.feature_importances_)

    # Prediction in sample
    RandomForests_PredInSample_y = FT.MachineLearning.Metrics.Predict_WithCorrectIndex(RandomForestModel, Df_X_Train)
    # Prediction in sample VS Observed
    #RandomForestsComp = pd.DataFrame(RandomForests_PredInSample_y).merge(Df_y_Train, how="inner", left_index=True, right_index=True)
    #RandomForestsComp.plot(style="o")
    # Prediction out sample
    RandomForests_PredOutSample_y = FT.MachineLearning.Metrics.Predict_WithCorrectIndex(RandomForestModel, Df_X_Test)

    #Create a dataframe to compare the observed values with all predicted in sample values
    AllModelComparaison_Prediction_InSample = Train_SimpleLReg_AllIn_ResultsInSample_Series[[IndependentVariable, "OLS Fitted Values"]]
    AllModelComparaison_Prediction_InSample = AllModelComparaison_Prediction_InSample.rename(columns={"OLS Fitted Values":"SimpleReg AllIn"})
    AllModelComparaison_Prediction_InSample = AllModelComparaison_Prediction_InSample.merge(Train_SimpleLReg_BackwardElimination_ResultsInSample_Series["OLS Fitted Values"], how="inner", left_index=True, right_index=True)
    AllModelComparaison_Prediction_InSample = AllModelComparaison_Prediction_InSample.rename(columns={"OLS Fitted Values":"SimpleReg BE"})
    AllModelComparaison_Prediction_InSample = AllModelComparaison_Prediction_InSample.merge(Train_PolyLReg_BackwardElimination_ResultsInSample_Series["OLS Fitted Values"], how="inner", left_index=True, right_index=True)
    AllModelComparaison_Prediction_InSample = AllModelComparaison_Prediction_InSample.rename(columns={"OLS Fitted Values":"PolyReg BE"})
    AllModelComparaison_Prediction_InSample = AllModelComparaison_Prediction_InSample.merge(pd.DataFrame(SVR_PredInSample_y), how="inner", left_index=True, right_index=True)
    AllModelComparaison_Prediction_InSample = AllModelComparaison_Prediction_InSample.rename(columns={0:"SVR"})
    AllModelComparaison_Prediction_InSample = AllModelComparaison_Prediction_InSample.merge(pd.DataFrame(DecisionTree_PredInSample_y), how="inner", left_index=True, right_index=True)
    AllModelComparaison_Prediction_InSample = AllModelComparaison_Prediction_InSample.rename(columns={0:"Decision Tree"})
    AllModelComparaison_Prediction_InSample = AllModelComparaison_Prediction_InSample.merge(pd.DataFrame(RandomForests_PredInSample_y), how="inner", left_index=True, right_index=True)
    AllModelComparaison_Prediction_InSample = AllModelComparaison_Prediction_InSample.rename(columns={0:"Random Forests"})

    # Graph of all predicted values and the observed values
    # AllModelComparaison_Prediction_InSample["x"] = AllModelComparaison_Prediction_InSample.index
    # AllModelComparaison_Prediction_InSample.plot(x="x", y=[IndependentVariable, "SimpleReg AllIn", "SimpleReg BE", "PolyReg BE", "Random Forests"], style=".")

    #Create a dataframe to compare the in sample residual series for each model
    AllModelComparaison_Residual_InSample = AllModelComparaison_Prediction_InSample.sub(AllModelComparaison_Prediction_InSample[IndependentVariable], axis=0)
    AllModelComparaison_Residual_InSample = AllModelComparaison_Residual_InSample.drop(IndependentVariable, axis=1)
    # Graph of the residual for each model
    # AllModelComparaison_Residual_InSample.reset_index().plot(x="index", y=["SimpleReg AllIn", "SimpleReg BE", "PolyReg BE", "Random Forests"], style=".")



    #Create a dataframe to compare the observed values with all predicted out sample values
    AllModelComparaison_Prediction_OutSample = Df_y_Test
    AllModelComparaison_Prediction_OutSample = AllModelComparaison_Prediction_OutSample.merge(SimpleLReg_AllIn_PredOutSample_y, how="inner", left_index=True, right_index=True)
    AllModelComparaison_Prediction_OutSample = AllModelComparaison_Prediction_OutSample.rename(columns={0:"SimpleReg AllIn"})
    AllModelComparaison_Prediction_OutSample = AllModelComparaison_Prediction_OutSample.merge(SimpleLReg_BackwardElimination_PredOutSample_y, how="inner", left_index=True, right_index=True)
    AllModelComparaison_Prediction_OutSample = AllModelComparaison_Prediction_OutSample.rename(columns={0:"SimpleReg BE"})
    AllModelComparaison_Prediction_OutSample = AllModelComparaison_Prediction_OutSample.merge(PolyLReg_BackwardElimination_PredOutSample_y, how="inner", left_index=True, right_index=True)
    AllModelComparaison_Prediction_OutSample = AllModelComparaison_Prediction_OutSample.rename(columns={0:"PolyReg BE"})
    AllModelComparaison_Prediction_OutSample = AllModelComparaison_Prediction_OutSample.merge(pd.DataFrame(SVR_PredOutSample_y), how="inner", left_index=True, right_index=True)
    AllModelComparaison_Prediction_OutSample = AllModelComparaison_Prediction_OutSample.rename(columns={0:"SVR"})
    AllModelComparaison_Prediction_OutSample = AllModelComparaison_Prediction_OutSample.merge(pd.DataFrame(DecisionTree_PredOutSample_y), how="inner", left_index=True, right_index=True)
    AllModelComparaison_Prediction_OutSample = AllModelComparaison_Prediction_OutSample.rename(columns={0:"Decision Tree"})
    AllModelComparaison_Prediction_OutSample = AllModelComparaison_Prediction_OutSample.merge(pd.DataFrame(RandomForests_PredOutSample_y), how="inner", left_index=True, right_index=True)
    AllModelComparaison_Prediction_OutSample = AllModelComparaison_Prediction_OutSample.rename(columns={0:"Random Forests"})

    # Graph of all predicted values and the observed values
    # AllModelComparaison_Prediction_OutSample["x"] = AllModelComparaison_Prediction_OutSample.index
    # AllModelComparaison_Prediction_OutSample.plot(x="x", y=[IndependentVariable, "SimpleReg AllIn", "SimpleReg BE", "PolyReg BE", "Random Forests"], style=".")

    #Create a dataframe to compare the out sample residual series for each model
    AllModelComparaison_Residual_OutSample = AllModelComparaison_Prediction_OutSample.sub(AllModelComparaison_Prediction_OutSample[IndependentVariable], axis=0)
    AllModelComparaison_Residual_OutSample = AllModelComparaison_Residual_OutSample.drop(IndependentVariable, axis=1)
    # Graph of the residual for each model
    # AllModelComparaison_Residual_OutSample.reset_index().plot(x="index", y=["SimpleReg AllIn", "SimpleReg BE", "PolyReg BE", "Random Forests"], style=".")

    return Models, \
           AllModelComparaison_Prediction_InSample, AllModelComparaison_Residual_InSample, \
           AllModelComparaison_Prediction_OutSample, AllModelComparaison_Residual_OutSample
