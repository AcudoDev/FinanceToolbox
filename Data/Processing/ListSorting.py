import pandas as pd
import numpy as np

import datetime

import openpyxl as pyxl
import re

def ListSorting_Alphanumeric(list):
    """
    Alphanumeric sorting for list object
    ----------
    Arguments:
        - list: list
            The list to sort

    Return:
        - list: list
            The inputed list sorted
    """

    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)]

    return sorted(list, key=alphanum_key)
