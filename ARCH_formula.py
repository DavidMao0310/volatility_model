import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style('whitegrid')
from pandas.plotting import lag_plot
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.gofplots import qqplot
from statsmodels.tsa.stattools import arma_order_select_ic
from statsmodels.tsa.arima_model import ARIMA
from arch import arch_model
from scipy.stats import shapiro
from scipy.stats import probplot
import armagarch as ag
from statsmodels.stats.diagnostic import het_arch
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox


def autocorr(x):
    plt.figure()
    lag_plot(x, lag=1, alpha=0.2, c='cornflowerblue')
    plt.title('Autocorrelation plot with lag = 1')
    plt.show()


# Stationary?
def look(x):
    plt.plot(x)
    plt.title('Return')
    plt.show()


def ADF(x):
    t = adfuller(x)
    output = pd.DataFrame(
        index=['Test Statistic Value', "p-value", "Lags Used", "Number of Observations Used", "Critical Value(1%)",
               "Critical Value(5%)", "Critical Value(10%)"], columns=['value'])
    output['value']['Test Statistic Value'] = t[0]
    output['value']['p-value'] = t[1]
    output['value']['Lags Used'] = t[2]
    output['value']['Number of Observations Used'] = t[3]
    output['value']['Critical Value(1%)'] = t[4]['1%']
    output['value']['Critical Value(5%)'] = t[4]['5%']
    output['value']['Critical Value(10%)'] = t[4]['10%']
    print(output)
    print('\n\n\n')


# We need bulid stationary series

def diff1(x):
    t = x.diff(1).dropna()
    return t


# Since P-Value small, we have enough evidence to say stationary
# White Noise test

def LBQ(x, lags=10):
    wn_pvalue = acorr_ljungbox(x, lags=lags, return_df=True)
    print("Ljung Box Q Test\n", wn_pvalue)
    if wn_pvalue['lb_pvalue'].values[-1] < 0.1:
        print('ARMA model is needed')
    else:
        print('ARMA model is not needed')

    print('\n\n\n')


# Since P-Value small, we have enough evidence to say non-WN

def archtest(x, lags=20):
    for i in range(5, lags, 5):
        arch_test = het_arch(x, maxlag=lags)
        print('ARCH test result\n (h, P-value, F_stat, Critical)=', arch_test)
        if arch_test[-1] < 0.1:
            print(str(i) + 'lag test GARCH model is needed')
        else:
            print(str(i) + 'lag test GARCH model is not needed')
    print('\n\n\n')


# To select ARIMA order
def look_order(t, lags=20):
    fig = plt.figure(figsize=(15, 5))
    plot_acf(t, lags=lags, ax=fig.add_subplot(121))
    plot_pacf(t, lags=lags, ax=fig.add_subplot(122))
    plt.show()


def auto_order(x, ar, ma):  # max_ar, max_ma
    z = arma_order_select_ic(x, max_ar=ar, max_ma=ma, ic='aic')
    print(z)
    print('\n\n\n')


# Build ARIMA Model
def arima_buildfit(x, order):
    model = ARIMA(x, order=order)
    t = model.fit()
    return t


# Analysis residuals
def tsdisplay(y, figsize=(14, 8), lags=10):
    tmp_data = pd.Series(y)
    fig = plt.figure(figsize=figsize)
    # Plot the time series
    tmp_data.plot(ax=fig.add_subplot(311), title="Time Series of Residuals", legend=False, c='forestgreen', alpha=0.7)
    # Plot the ACF:
    plot_acf(tmp_data, lags=lags, zero=False, ax=fig.add_subplot(323))
    plt.xticks(np.arange(1, lags + 1, 1.0))
    # Plot the PACF:
    plot_pacf(tmp_data, lags=lags, zero=False, ax=fig.add_subplot(324))
    plt.xticks(np.arange(1, lags + 1, 1.0))
    # Plot the QQ plot of the data:
    qqplot(tmp_data, line='s', ax=fig.add_subplot(325), c='cornflowerblue')
    plt.title("QQ Plot")
    # Plot the residual histogram:
    fig.add_subplot(326).hist(tmp_data, bins=40, density=True, range=[-5, 5])
    plt.title("Histogram")
    # Fix the layout of the plots:
    plt.tight_layout()
    plt.show()


def arimapredict(fit, oridata,name='required'):
    fig = plt.figure(figsize=(12, 8))
    data_size = oridata.shape[0]
    fit.plot_predict(1, data_size - 20, ax=fig.add_subplot(211), alpha=0.05, plot_insample=True)
    plt.title(str(name)+' return prediction')
    plt.legend().remove()
    fit.plot_predict(data_size - 30, data_size + 20, ax=fig.add_subplot(212))
    plt.tight_layout()
    plt.show()

def make_arima(df,order=(1,1,1),name='required'):
    modelfit = arima_buildfit(df, (order[0], order[1], order[2]))
    print(modelfit.summary())
    arima_residuals = modelfit.resid
    tsdisplay(arima_residuals)
    arimapredict(modelfit, df, name=name)


def diagnose(resi, lbqlags=15):
    print('Ljung Box Test Check \n')
    lbqtest = acorr_ljungbox(resi, lags=lbqlags, return_df=True)
    print("Ljung Box Q Test\n", lbqtest)
    print('ARCH Test Check \n')
    for i in range(5, lbqlags+5, 5):
        arch_test = het_arch(resi, maxlag=20)
        print('ARCH test result\n (h, P-value, F_stat, Critical)=', arch_test)


def make_armagarch(df,armaorder=(1,1),garchorder=(1,1),dist='T',name='required'):
    meanMdl = ag.ARMA(order={'AR': armaorder[0], 'MA': armaorder[1]})
    volMdl = ag.garch(order={'p': garchorder[0], 'q': garchorder[1]})
    if dist=='T':
        dist = ag.tStudent()
    elif dist=='N':
        dist = ag.normalDist()
    else:
        print('Error')
        pass

    # set-up the model
    model = ag.empModel(df.to_frame(), meanMdl, volMdl, dist)
    model.fit()
    # get the conditional mean
    Ey = model.Ey
    # get conditional variance
    ht = model.ht
    cvol = np.sqrt(ht)
    # get standardized residuals
    stres = model.stres
    diagnose(stres,lbqlags=15)
    # plot in three subplots
    fig, ax = plt.subplots(3, 1, figsize=(13, 10))
    Ey.plot(ax=ax[0], color='forestgreen', alpha=0.7, title='Expected returns', legend=False)
    cvol.plot(ax=ax[1], color='blue', title='Conditional volatility', legend=False)
    stres.plot(ax=ax[2], color='sandybrown', title='Standardized residuals', legend=False)
    plt.subplots_adjust(hspace=0.5)
    fig.tight_layout()
    plt.show()
    # make a prediction of mean and variance over next 3 days.
    pred = model.predict(nsteps=59)
    predvol=pred[1]
    predvol=pd.DataFrame(predvol, columns=['predict_volatility'],index=pd.date_range(start='20210331', end='20210528'))
    fig = plt.figure(figsize=(12, 8),)
    cvol['returnVol'].plot(color='steelblue',alpha=0.7)
    predvol['predict_volatility'].plot(color='orange')
    plt.legend()
    plt.title(str(name)+' volatility prediction')
    fig.tight_layout()
    plt.show()