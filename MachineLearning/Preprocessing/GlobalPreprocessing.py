import FinanceToolbox as FT


def GlobalPreprocessing(df,
                        Split_x_y=False, IndependentColumnName=None,
                        EncodeCategoricalVariable=True,
                        Normalize=True,
                        Split_Train_Test=False, TestSetSize=0.2, Randomize=True,
                        AddConst=False,
                        SortIndex=False):
    """
    This function allow to apply preprocessing tasks on a pandas dataframe with one line of code:
        - Split into independent and dependent sets
        - Split into train and test sets
        - Encode Categorical Variable
        - Normalize

    Arguments:
    ----------
        - df: pandas dataframe
            - The dataframe to treat
        - Split_x_y: boolean
            - if True, the dataframe is splitted into two subsets: independent and dependent variables. IndependentColumnName argument becomes compulsory.
            - if False, do nothing
        - IndependentColumnName: list
            Compulsory if Split_x_y is True. This is the list of one or moere independent variables column names
        - EncodeCategoricalVariable: boolean
            - if True, encode all categorical variables found and remove one column for each categorical variable found to avoid colinearity issues.
            - if False, do nothing
            If Split_x_y = True, then encoding is applied to both x and y
        - Normalize: boolean
            - if True, the dataframe is normalized (df-df.min())/(df.max()-df.min())
            - if False, do nothing
            If Split_x_y = True, then normalization is applied both to x and y
        - Split_Train_Test: boolean
            - if True, then the dataframe is splitted into train and test subsets.
            - if False, do nothing
            If Split_x_y = True, then both x and y is splitted.
        - TestSetSize: float
            The size of the test subset as percentage of the original dataframe size
            Default: 0.2
        - Randomize: boolean
            - if True, elements are randomly splitted between the train and test subsets
            - if False, always the same splitting into train and test subsets, to use for debug/testing purposes
            Default: True
        - AddConst : boolean
            - if True, Add a column at the index 0 filled with 1. Useful for adding a "constant" column
            - if False, do nothing
            Default = False
        - SortIndex: boolean
            Sort the index of the output data
            Default = False
    Return:
    ----------
        - if Split_x_y = True
            - if Split_Train_Test = True
                - Return: x_train, x_test, y_train, y_test
            - else
                - Return: x, y
        - else
            - if Split_Train_Test = True
                return df_train, df_test
            - else
                return df
        Note that "df" is the original dataframe. when df is returned, it's the original dataframe with treatment applied, i.e. normalization, adding constan column, etc.

    """
    if Split_x_y == True:
        x, y = FT.MachineLearning.Preprocessing.SplitDataset_x_y(df, IndependentColumnName)

    if EncodeCategoricalVariable == True:

        if Split_x_y == True:
            x = FT.MachineLearning.Preprocessing.EncodeCategoricalVariable(x)
            y = FT.MachineLearning.Preprocessing.EncodeCategoricalVariable(y)
        else:
            df = FT.MachineLearning.Preprocessing.EncodeCategoricalVariable(df)

    if Normalize == True:

        if Split_x_y == True:
            x = FT.MachineLearning.Preprocessing.Normalize(x)
            y = FT.MachineLearning.Preprocessing.Normalize(y)
        else:
            df = FT.MachineLearning.Preprocessing.Normalize(df)

    if AddConst == True:
        if Split_x_y == True:
            x.insert(0, "Const", 1)
        else:
            df.insert(0, "Const", 1)

    if Split_Train_Test == True:

        if Split_x_y == True:
            x_train, x_test, y_train, y_test = FT.MachineLearning.Preprocessing.SplitDataset_Train_Test(x, y, TestSetSize=TestSetSize, Randomize=Randomize)
        else:
            df_train, df_test = FT.MachineLearning.Preprocessing.SplitDataset_Train_Test(x=df, y=None, TestSetSize=TestSetSize, Randomize=Randomize)

    if SortIndex == True:
        if Split_x_y == True:
            if Split_Train_Test == True:
                x_train = x_train.sort_index()
                x_test = x_test.sort_index()
                y_train = y_train.sort_index()
                y_test = y_test.sort_index()
            else:
                x = x.sort_index()
                y = y.sort_index()
        else:
            if Split_Train_Test == True:
                df_train = df_train.sort_index()
                df_test = df_test.sort_index()
            else:
                df = df.sort_index()

    if Split_x_y == True:
        if Split_Train_Test == True:
            return x_train, x_test, y_train, y_test
        else:
            return x, y
    else:
        if Split_Train_Test == True:
            return df_train, df_test
        else:
            return df

