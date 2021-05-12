import pandas as pd
import numpy as np
import math
import talib as ta
import scipy.optimize as sco
from matplotlib import pyplot as plt
import VaR
plt.style.use('fivethirtyeight')


def read(df):
    df['Date'] = pd.to_datetime(df['Date'])
    df.set_index('Date', inplace=True)
    return df


full_df = pd.read_csv('dataset/test_portfolio.csv')
full_df = read(full_df)
riskfree = pd.read_csv('dataset/riskfree.csv')
riskfree = read(riskfree)
riskfree = riskfree.loc[full_df.index].dropna()
r_f = ta.EMA(riskfree['rate'].values, timeperiod=30)[-1]
# Resample the full dataframe to monthly timeframe
monthly_df = full_df.resample('BMS').first()
# Calculate daily returns of stocks
returns_daily = full_df.pct_change()
returns_daily.cumsum().plot(cmap='coolwarm', figsize=(10, 6))
plt.show()
# Calculate monthly returns of the stocks
returns_monthly = monthly_df.pct_change().dropna()
rets = returns_daily.dropna()


def get_annual(df):
    print('annualized return\n', df.mean() * 252)
    print('annualized volatility\n', df.std() * 252)


get_annual(rets)
rets = returns_daily.dropna()

###############################################################
nums = len(rets.columns.tolist())
weights = np.random.random(nums)
weights /= np.sum(weights)


def lookannualized(df, weights):
    prets = np.sum((weights * df.mean()) * 252)
    pstd = math.sqrt(np.dot(weights.T, np.dot(df.cov() * 252, weights)))
    table = pd.DataFrame(np.array([[prets, pstd]]),
                         columns=['Portfolio Annualized Return', 'Portfolio Annualized Volatility'])
    return table


lookannualized(rets, weights)


def gen_prets(weights):
    return np.sum((weights * rets.mean()) * 252)


def gen_pstd(weights):
    return math.sqrt(np.dot(weights.T, np.dot(rets.cov() * 252, weights)))


def negative_ret(weights):
    return -np.sum((weights * rets.mean()) * 252)


def gen_negative_pstd(weights):
    retss = rets[rets > 0]
    return math.sqrt(np.dot(weights.T, np.dot(retss.cov() * 252, weights)))


##Monte Carlo method
prets = []
pstd = []
for i in range(2000):
    weights = np.random.random(nums)
    weights /= np.sum(weights)
    prets.append(gen_prets(weights) - r_f)
    pstd.append(gen_pstd(weights))
prets = np.array(prets)
pstd = np.array(pstd)

fig = plt.figure(figsize=(12, 6))
plt.scatter(pstd, prets, c=(prets) / pstd,
            marker='o', cmap='viridis')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
fig.tight_layout()
plt.show()


def negative_sharpe(weights):
    return - (gen_prets(weights) - r_f) / gen_pstd(weights)


def negative_sortino(weights):
    return - (gen_prets(weights) - r_f) / gen_negative_pstd(weights)


cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
bnds = tuple((0, 1) for x in range(nums))
eweights = np.array(nums * [1. / nums, ])
opts = sco.minimize(negative_sharpe, eweights, method='SLSQP', bounds=bnds, constraints=cons)
bestsharpedf = pd.DataFrame(np.array([[gen_prets(opts['x']), gen_pstd(opts['x']),
                                       (gen_prets(opts['x']) - r_f) / gen_pstd(opts['x']),
                                       (gen_prets(opts['x']) - r_f) / gen_negative_pstd(opts['x'])]]),
                            columns=['Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio'],
                            index=['best sharpe ratio'])
weight_mem = []
weight_mem.append(opts['x'])

# volatility lowest
optv = sco.minimize(gen_pstd, eweights, method='SLSQP', bounds=bnds, constraints=cons)
lowestvoladf = pd.DataFrame(np.array([[gen_prets(optv['x']), gen_pstd(optv['x']),
                                       (gen_prets(optv['x']) - r_f) / gen_pstd(optv['x']),
                                       (gen_prets(optv['x']) - r_f) / gen_negative_pstd(optv['x'])]]),
                            columns=['Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio'],
                            index=['lowest volatility'])

weight_mem.append(optv['x'])
# max return
optr = sco.minimize(negative_ret, eweights, method='SLSQP', bounds=bnds, constraints=cons)
maxretdf = pd.DataFrame(np.array([[gen_prets(optr['x']), gen_pstd(optr['x']),
                                   (gen_prets(optr['x']) - r_f) / gen_pstd(optr['x']),
                                   (gen_prets(optr['x']) - r_f) / gen_negative_pstd(optr['x'])]]),
                        columns=['Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio'], index=['Highest Return'])

weight_mem.append(optr['x'])
# best sortino
optso = sco.minimize(negative_sortino, eweights, method='SLSQP', bounds=bnds, constraints=cons)
maxsort = pd.DataFrame(np.array([[gen_prets(optso['x']), gen_pstd(optso['x']),
                                  (gen_prets(optso['x']) - r_f) / gen_pstd(optso['x']),
                                  (gen_prets(optso['x']) - r_f) / gen_negative_pstd(optso['x'])]]),
                       columns=['Return', 'Volatility', 'Sharpe Ratio', 'Sortino Ratio'], index=['Best Sortino Ratio'])

weight_mem.append(optso['x'])
result = pd.concat([bestsharpedf, lowestvoladf, maxretdf, maxsort])
print(weight_mem)
print(result)
# latest_efficient_frontie

bnds = tuple((0, 1) for x in eweights)

frets = np.linspace(0.1, 0.13, 100)
fstd = []
for fret in frets:
    cons = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: gen_prets(x) - fret})
    optf = sco.minimize(gen_pstd, eweights, method='SLSQP',
                        bounds=bnds, constraints=cons)
    fstd.append(optf['fun'])
fstd = np.array(fstd)

fig = plt.figure(figsize=(12, 6))
plt.scatter(pstd, prets, c=(prets) / pstd,
            marker='o', cmap='viridis')
plt.plot(fstd, frets, 'b')
plt.plot(gen_pstd(opts['x']), gen_prets(opts['x']), marker='*',
         markersize=20.0, label='bset sharpe ratio', color='red')
plt.plot(gen_pstd(optv['x']), gen_prets(optv['x']), marker='*',
         markersize=20.0, label='lowest volatility', color='coral')
plt.plot(gen_pstd(optr['x']), gen_prets(optr['x']), marker='*',
         markersize=20.0, label='highest return', color='pink')
plt.xlabel('expected volatility')
plt.ylabel('expected return')
plt.colorbar(label='Sharpe ratio')
plt.legend()
fig.tight_layout()
plt.show()



tdata = pd.read_csv('dataset/test_portfolio.csv', header=0, index_col=0)
VaR.DoVaR(tdata,weight_mem[1])