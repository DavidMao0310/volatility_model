import numpy as np
import pandas as pd
import talib as ta
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')
sp = pd.read_csv('dataset/SP500_futures_15yr.csv')
nasdaq = pd.read_csv('dataset/Nasdaq_futures_15yr.csv')
dj = pd.read_csv('dataset/DownJ_futures_15yr.csv')
def preprocess(df):
    df['return'] = df['Close'].pct_change(1).round(4)
    df['squared_daily_return'] = np.square(df['return'])
    df['absolute_daily_return'] = np.abs(df['return'])
    df['original_volatility'] = ta.STDDEV(df['return'].values,timeperiod=10)
    df['EMA_volatility'] = ta.EMA(df['original_volatility'].values,timeperiod=30)
    df['log_return'] = np.log(1 + df['Close'].pct_change(1).round(4))
    df.set_index('Date',inplace=True)
    df = df.set_index(pd.to_datetime(df.index))
    df.dropna(inplace=True)
    return df

def ewma(data,b=0.94,name='expected'):
    data = preprocess(data)
    use_data = data['squared_daily_return']
    u60 = use_data[-59:]
    vol_ewma = np.zeros(58)
    vol_ewma[0] = np.array(use_data)[-117:-58].std()
    for i in range(57):
        vol_ewma[i + 1] = np.sqrt((1 - b) * np.array(u60)[i] + b * vol_ewma[i] ** 2)
    predict = pd.DataFrame(vol_ewma, columns=['predict_volatility'], index=pd.date_range(start='20210331', end='20210527'))
    fig = plt.figure(figsize=(12, 8),)
    data['original_volatility'].plot(alpha=0.5)
    data['EMA_volatility'].plot(color='red',alpha=0.5)
    predict['predict_volatility'].plot(color='orange')
    plt.legend()
    plt.title('EWMA Model for '+str(name))
    fig.tight_layout()
    plt.show()

ewma(data=sp,b=0.94,name='S&P500_futures')
ewma(data=nasdaq,b=0.94,name='Nasdaq_futures')
ewma(data=dj,b=0.94,name='DowJones_futures')







