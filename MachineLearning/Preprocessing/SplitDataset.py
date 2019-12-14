import pandas as pd
from sklearn.model_selection import train_test_split


def SplitDataset_Train_Test(x, y, TestSetSize=0.2, Randomize=True):
    """
    This function split the dataset into train and test subsets.

    Arguments:
    ----------
        - x: pandas dataframe
            The dataframe containing the explanatory variables
        - y: pandas dataframe
            The dataframe containing the independent variable(s)
        - TestSetSize: float
            The size of the test subset in percentage - from 0 to 1.
            As a consequence, the train set size is 1 - the test set size.
            Default= 0.2
        - Randomize: boolean
            - if True, elements are randomly splitted between the train and test subsets
            - if False, always the same splitting, to use for debug/testing purposes
            Default: True
    Return:
    ----------
        - x_train: pandas dataframe
        - x_test: pandas dataframe
        - y_train: pandas dataframe
        - y_test: pandas dataframe

        All this variables are respectively for x or y initial set divided into ttrain and test subsets.

    """
    if Randomize == False:
        if y == None:
            return train_test_split(x, test_size=TestSetSize, random_state=0)
        else:
            return train_test_split(x, y, test_size=TestSetSize, random_state=0)
    else:
        if y == None:
            return train_test_split(x, test_size=TestSetSize)
        else:
            return train_test_split(x, y, test_size=TestSetSize)

def SplitDataset_x_y(df, IndependentColumnName):
    """
    This function split a dataframe into a dataframe with independent variables and a dataframe with dependent variables.
    The IndependentColumnName define what is/are the independent variables based on the related column name.

    Arguments:
    ----------
        - df: pandas dataframe
            The dataframe containing both the independent and dependent variables
        - IndependentColumnName: list
            The column name of the independent variables - can be a list of one element or more.

    Return:
    ----------
        - x: pandas dataframe
            All columns contained in the initial df dataframe excepted the column provided into the IndependentColumnName list.
        - y: pandas dataframe
            Only the columns provided into the IndependentColumnName

    """
    y = df[IndependentColumnName]
    x = df.drop(IndependentColumnName, axis=1)

    return x, y