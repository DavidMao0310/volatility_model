import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import preprocessing as pre
import talib as ta
import arch

pd.set_option('display.max_columns', None)
plt.style.use('fivethirtyeight')

data = pd.read_excel('dataset/TestData.xlsx')
sp500_futures = pd.read_csv('dataset/SP500_futures_15yr.csv')
dj100_futures = pd.read_csv('dataset/DownJ_futures_15yr.csv')
nasdaq_futures = pd.read_csv('dataset/Nasdaq_futures_15yr.csv')


def preprocess(df):
    df['return'] = df['Close'].pct_change(1).round(4)
    df['squared_daily_return'] = np.square(df['return'])
    df['absolute_daily_return'] = np.abs(df['return'])
    df.set_index('Date',inplace=True)
    df = df.set_index(pd.to_datetime(df.index))
    df.dropna(inplace=True)
    return df

sp500_futures = preprocess(sp500_futures)
dj100_futures = preprocess(dj100_futures)
nasdaq_futures = preprocess(nasdaq_futures)

plt.figure(figsize=(11,8))
sp500_futures['Close2']=np.array(pre.MinMaxScaler().fit_transform(sp500_futures[['Close']]))
dj100_futures['Close2']=np.array(pre.MinMaxScaler().fit_transform(dj100_futures[['Close']]))
nasdaq_futures['Close2']=np.array(pre.MinMaxScaler().fit_transform(nasdaq_futures[['Close']]))
sp500_futures['Close2'].plot(label='S&P500_futures')
dj100_futures['Close2'].plot(label='Dowjones_futures')
nasdaq_futures['Close2'].plot(label='Nasdaq_futures')
plt.title('Scaled Close price')
plt.legend()
plt.show()


def show_basic_plot(data,name):
    fig = plt.figure(figsize=(14, 11),)
    ax1 = fig.add_subplot(311)
    data['Close'].plot(color='blue')
    plt.title(str(name)+'Close Price')
    ax2 = fig.add_subplot(312)
    data['squared_daily_return'].plot(color='forestgreen')
    plt.title(str(name)+'Squared daily return')
    ax3 = fig.add_subplot(313)
    data['absolute_daily_return'].plot(color='sandybrown')
    plt.title(str(name)+'Absolute daily return')
    fig.tight_layout()
    plt.show()

show_basic_plot(sp500_futures,'S&P500_futures')
show_basic_plot(nasdaq_futures,'Nasdaq_futures')
show_basic_plot(dj100_futures,'Dowjones_futures')

