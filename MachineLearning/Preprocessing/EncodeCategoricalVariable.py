import pandas as pd

def EncodeCategoricalVariable(df, prefix_sep="_"):
    """
    This function encode all categorical variables in a pandas dataframe.
    Each categorical variables is encoded and the first column (of each categorical variable encoded) is removed to avoid colinearity issues.

    Argument
    ----------
        - df: Pandas Dataframe
            The dataframe to encode
        - prefix_sep: str
            The the separator to put between the original column name and each category. i.e Gender_Male, Gender_Female if you choosed "_" as separator.
            Default: _

    Return:
    ----------
        - df: pandas dataframe
            The encoded dataframe

    """
    return pd.get_dummies(df, prefix_sep=prefix_sep, drop_first=True)