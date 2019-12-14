import pandas as pd

def Normalize(df):
    """
    This function takes a pandas dataframe and normalize it

    Arguments
    ----------
        - df: pandas dataframe

    Return
    ----------
        - df: pandas dataframe
            The initial dataframe normalized

    """
    df = (df - df.min())/(df.max() - df.min())
    return df