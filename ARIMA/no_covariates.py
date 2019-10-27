%matplotlib inline
import pandas as pd
import numpy as np
from pandas import datetime

def parser(x):
    d = datetime.strptime(x, '%b %d, %Y')
    return d


data = pd.read_csv("Nikkei.csv", index_col=0, usecols=["Date", "Price"], parse_dates=[0], date_parser=parser)
data = data.reindex(index=data.index[::-1])

# data_2 = pd.read_csv("Crude_Oil_monthly_10.csv", index_col=0, usecols=["Date", "Price_Oil"], parse_dates=[0], date_parser=parser)
# data_2 = data_2.reindex(index=data_2.index[::-1])

print(data.head())
# print(data_2.head())
