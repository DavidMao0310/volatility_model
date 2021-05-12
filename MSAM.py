import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt
from arch.unitroot import ADF

plt.style.use('fivethirtyeight')

data = pd.read_csv('dataset/SP500_futures_15yr.csv')
def datapre(df):
    df.index = pd.to_datetime(df.Date)
    df_ret = df.Close.resample('W').last().pct_change().dropna()
    return df_ret

def make_MASM(data,name):
    df_ret = datapre(data)
    df_ret.plot(title=str(name)+'_Weekly Return', figsize=(15, 4),color='forestgreen')
    plt.show()
    # ADF Test
    adf = ADF(df_ret)
    print('ADF test result \n', adf)
    # Fit MSAM model
    mod = sm.tsa.MarkovRegression(df_ret.dropna(),
                                  k_regimes=3, trend='nc', switching_variance=True)
    res = mod.fit()
    print(res.summary())

    fig, axes = plt.subplots(3, figsize=(12, 8))
    ax = axes[0]
    ax.plot(res.smoothed_marginal_probabilities[0],color='skyblue')
    ax.set(title=str(name)+'_Low volatility smoothed probability graph')
    ax = axes[1]
    ax.plot(res.smoothed_marginal_probabilities[1],color='darkseagreen')
    ax.set(title=str(name)+'_Middle volatility smoothed probability graph')
    ax = axes[2]
    ax.plot(res.smoothed_marginal_probabilities[2],color='sandybrown')
    ax.set(title=str(name)+'_High volatility smoothed probability graph')
    fig.tight_layout()
    plt.show()


make_MASM(data,name='S&P500_futures')



