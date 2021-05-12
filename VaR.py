import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as st
import math
pd.set_option('display.max_columns', None)
plt.style.use('fivethirtyeight')




# Use Delta_Normal get VaR
def VaR_VCM(Value, Rp, Vp, X, N):
    '''Value：value of portfolio；
       Rp:portfolio expected return;
       Vp:portfolio daily volatility;
       X:confidence level；
       N:holding period'''
    z = np.abs(st.norm.ppf(q=1 - X))
    return abs(np.sqrt(N) * Value * (Rp - z * Vp))


# Use Delta_Normal get ES
def ES_VCM(Value, Rp, Vp, X, N):
    '''Value：value of portfolio；
       Rp:portfolio expected return;
       Vp:portfolio daily volatility;
       X:confidence level；
       N:holding period'''
    z = np.abs(st.norm.ppf(q=1 - X))
    return abs(Value * (Rp - Vp * (math.exp(-(z ** 2 / 2))) / (np.sqrt(2 * math.pi) * (1 - X))) * np.sqrt(N))


def VaR_history(a, q):
    VaR = np.percentile(a, (1 - q) * 100)
    return abs(VaR)


def ES_history(a, q):
    VaR = np.percentile(a, (1 - q) * 100)
    ES = a[a <= VaR].mean()
    return abs(np.array(ES)[0])




data = pd.read_csv('dataset/test_portfolio.csv', header=0, index_col=0)
weights = np.array([0.40, 0.30, 0.30])

def DoVaR(data,weights):
    R = np.log(data / data.shift(1))  # get log return
    R = R.dropna()
    # Get cov
    R_cov = R.cov()
    R_corr = R.corr()
    sns.heatmap(R_corr, cmap='coolwarm')
    plt.show()
    R_mean = R.mean()
    R_vol = R.std()
    Rp_daily = np.sum(weights * R_mean)
    Vp_daily = np.sqrt(np.dot(weights, np.dot(R_cov, weights.T)))
    print('Average daily rate of return of the portfolio', round(Rp_daily, 6))
    print('Daily volatility of the portfolio', round(Vp_daily, 6))

    def delatnormal_result(Value_port=1000000):
        D1 = 1
        D2 = 10
        X1 = 0.99
        X2 = 0.95
        Value_port = Value_port

        VaR99_1day_VCM = VaR_VCM(Value=Value_port, Rp=Rp_daily, Vp=Vp_daily, X=X1, N=D1).round(2)
        VaR99_1day_ES = ES_VCM(Value=Value_port, Rp=Rp_daily, Vp=Vp_daily, X=X1, N=D1).round(2)

        VaR95_1day_VCM = VaR_VCM(Value=Value_port, Rp=Rp_daily, Vp=Vp_daily, X=X2, N=D1).round(2)
        VaR95_1day_ES = ES_VCM(Value=Value_port, Rp=Rp_daily, Vp=Vp_daily, X=X2, N=D1).round(2)

        VaR99_10day_VCM = VaR_VCM(Value=Value_port, Rp=Rp_daily, Vp=Vp_daily, X=X1, N=D2).round(2)
        VaR99_10day_ES = ES_VCM(Value=Value_port, Rp=Rp_daily, Vp=Vp_daily, X=X1, N=D2).round(2)

        VaR95_10day_VCM = VaR_VCM(Value=Value_port, Rp=Rp_daily, Vp=Vp_daily, X=X2, N=D2).round(2)
        VaR95_10day_ES = ES_VCM(Value=Value_port, Rp=Rp_daily, Vp=Vp_daily, X=X2, N=D2).round(2)
        table = pd.DataFrame(np.array([[VaR99_1day_VCM, VaR99_10day_VCM], [VaR99_1day_ES, VaR99_10day_ES],
                                       [VaR95_1day_VCM, VaR95_10day_VCM], [VaR95_1day_ES, VaR95_10day_ES]]),
                             columns=['Delta_Normal 1 day', 'Delta_Normal 10 days'],
                             index=['99%_VaR', '99%_ES', '95%_VaR', '95%_ES'])
        return table

    def history_result(Value_port=1000000):
        Value_port = Value_port

        Value_asset = Value_port * weights
        Return_history = np.dot(R, Value_asset)
        Return_history = pd.DataFrame(Return_history, index=R.index, columns=['Portfolio Daily Return'])
        Return_history.describe()
        Return_history.plot(figsize=(8, 5))

        plt.figure(figsize=(8, 6))
        plt.hist(np.array(Return_history), bins=30, facecolor='y', edgecolor='k')
        plt.xlabel(u'Daily income amount of the portfolio', fontsize=13)
        plt.ylabel(u'Frequency', fontsize=13, rotation=0)
        plt.xticks(fontsize=13)
        plt.yticks(fontsize=13)
        plt.title(u'Histogram of portfolio daily return rate', fontsize=13)
        plt.show()

        stt = st.kstest(rvs=Return_history['Portfolio Daily Return'], cdf='norm')
        print(stt)
        VaR99_1day_history = VaR_history(a=Return_history, q=0.99).round(2)
        ES99_1day_history = ES_history(a=Return_history, q=0.99).round(2)

        VaR95_1day_history = VaR_history(a=Return_history, q=0.95).round(2)
        ES95_1day_history = ES_history(a=Return_history, q=0.95).round(2)

        VaR99_10day_history = (np.sqrt(10) * VaR99_1day_history).round(2)
        ES99_10day_history = (np.sqrt(10) * ES99_1day_history).round(2)

        VaR95_10day_history = (np.sqrt(10) * VaR95_1day_history).round(2)
        ES95_10day_history = (np.sqrt(10) * ES95_1day_history).round(2)

        table = pd.DataFrame(
            np.array([[VaR99_1day_history, VaR99_10day_history], [ES99_1day_history, ES99_10day_history],
                      [VaR95_1day_history, VaR95_10day_history], [ES95_1day_history, ES95_10day_history]]),
            columns=['Historical simulation 1 day', 'Historical simulation 10 days'],
            index=['99%_VaR', '99%_ES', '95%_VaR', '95%_ES'])
        return table

    final_result = pd.concat([delatnormal_result(1000000), history_result(1000000)], axis=1)
    print(final_result)

DoVaR(data,weights)

