import pandas as pd
import numpy as np

sp = pd.read_csv('dataset/SP500_futures_15yr.csv')
nasdaq = pd.read_csv('dataset/Nasdaq_futures_15yr.csv')
dj = pd.read_csv('dataset/DownJ_futures_15yr.csv')

def makedata(df,name):
    df.set_index('Date', inplace=True)
    df = df.set_index(pd.to_datetime(df.index))
    df[str(name)]=df['Close']
    df = df[str(name)]
    return df

sp = makedata(sp,'S&P500')
nasdaq = makedata(nasdaq,'Nasdaq')
dj = makedata(dj,'DowJones')

data = pd.concat([sp,nasdaq,dj],axis=1)
print(data)
data.to_csv('dataset/test_portfolio.csv')