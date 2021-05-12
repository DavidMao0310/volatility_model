import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import talib as ta
import arch

plt.style.use('ggplot')

data = pd.read_excel('TestData.xlsx')
data['returnSP'] = data['SP'].pct_change(1).round(4)
data['volSQR'] = np.square(data['returnSP'])
data['volABS'] = np.abs(data['returnSP'])
print(data)


def show_basic_plot():
    fig = plt.figure(figsize=(18, 13))
    ax1 = fig.add_subplot(311)
    data['SP'].plot()
    plt.title('PriceSP')
    ax2 = fig.add_subplot(312)
    data['returnSP'].plot()
    plt.title('returnSP')
    ax3 = fig.add_subplot(313)
    data['volSQR'].plot()
    plt.title('volSQR')
    plt.show()

# show_basic_plot()
