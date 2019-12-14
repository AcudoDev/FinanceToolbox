import pandas as pd

def ConvertToPolynomial(df, degrees):
    """
    This function convert a dataframe of variables to its polynomial equivalence

    Argument:
    ----------
        - df: pandas dataframe
            The dataframe to convert
        - degress: list
            The list of degrees to generate. Providing [1, 2, 3] for df only containing a column X, output X, X^2, X^3.
            NOTE: if you don't put 1 in the list, you loose the original column (i.e. X^1)

    Return:
    ----------
        - df_poly: pandas dataframe
            the dataframe converted in the polynomial form
    """

    InitialColumns = df.columns

    for col in InitialColumns:
        for degree in degrees:
                df[col + "_^" + str(degree)] = df[col] ** degree

    return df