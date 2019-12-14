import csv
import pandas as pd

def FromCSVToPandasDataFrame(File):
    df = pd.read_csv(File)
    return df
file = r".\SwapFlyAnalyzer\settings.csv"
print(FromCSVToPandasDataFrame(file))