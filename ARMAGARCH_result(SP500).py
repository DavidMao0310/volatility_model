import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ARCH_formula as fun
plt.style.use('fivethirtyeight')

data = pd.read_csv('dataset/SP500_futures_15yr.csv')
def datapre(data):
    data['return'] = data['Close'].pct_change(1)
    data.set_index('Date', inplace=True)
    data = data.set_index(pd.to_datetime(data.index))
    data.dropna(inplace=True)
    data = 100 * data['return']
    return data
data = datapre(data)
print(data)


#Stationary?
#fun.look(data)

print('ADF Test \n')
fun.ADF(data)

print('Ljung Box Test \n')
#WN test ARMA is needed?
fun.LBQ(data,lags=15)
fun.look_order(data,lags=20)
#fun.auto_order(data,5,5)
#ARCH test, GARCH is needed?
fun.archtest(data,lags=20)
#fit ARMA model
fun.make_arima(data,order=(2,0,1),name='S&P500_futures')
#fit_ARMAGARCH
fun.make_armagarch(data,armaorder=(2,1),garchorder=(1,1),
                   dist='T',name='S&P500_futures')





