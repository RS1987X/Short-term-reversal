# -*- coding: utf-8 -*-
"""
Created on Wed Jun  7 12:20:31 2023

@author: Richard
"""

"""

Buying stock that is liquidated

"""


import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import math
from datetime import date

# get prices from yahoo finance

tday = date.today()
tday_str = tday.strftime("%Y-%m-%d")


data = pd.read_csv(r".\universe mcap above 1 mdr SEK 20230607.csv",delimiter=";")
 
yf_names = data["Yahoo finance name"]

tickers = yf_names.str.cat(sep=" ")


# =============================================================================
# ============================================================================
hist = yf.download(tickers, start='2019-01-01', end=tday_str)
# ============================================================================
# =============================================================================

# .dropna(how='all',inplace = True)#.fillna(0)
close_prices = hist["Close"]
close_prices.dropna(axis=0, how='all', inplace=True)
close_prices.ffill(axis=0,inplace=True)

open_prices = hist["Open"]
open_prices.dropna(axis=0, how='all', inplace=True)
open_prices.ffill(axis=0,inplace=True)
    
#close_prices = close_prices.drop([pd.Timestamp('2023-05-31 00:00:00')])#, pd.Timestamp(
 #   '2018-06-22 00:00:00'), pd.Timestamp('2017-06-23 00:00:00'), pd.Timestamp('2017-06-06 00:00:00')])
high_prices = hist["High"]
high_prices.ffill(axis=0,inplace=True)
low_prices = hist["Low"]
low_prices.ffill(axis=0,inplace=True)
#high_prices = high_prices.drop([pd.Timestamp('2018-06-06 00:00:00'), pd.Timestamp(
#    '2018-06-22 00:00:00'), pd.Timestamp('2017-06-23 00:00:00'), pd.Timestamp('2017-06-06 00:00:00')])

volumes = hist["Volume"].dropna(how='all')  # .fillna(0)
volumes.dropna(axis=0, how='all', inplace=True)
#volumes = volumes.drop([pd.Timestamp('2018-06-06 00:00:00'), pd.Timestamp('2018-06-22 00:00:00'),
#                       pd.Timestamp('2017-06-23 00:00:00'), pd.Timestamp('2017-06-06 00:00:00')])

avg_volume = volumes.shift(1).rolling(10).mean()
ret = close_prices/close_prices.shift(1)-1
ret_5d = close_prices.shift(1)/close_prices.shift(6)-1
ret_20d = close_prices.shift(1)/close_prices.shift(21)-1

consecutive_neg_returns = (ret.shift(1) < 0) & (ret.shift(2) < 0) & (ret.shift(3) < 0)
big_downday = (ret < -0.15)
high_volume = volumes > 5*avg_volume
close_high = (high_prices - close_prices) < 0.1*(high_prices - low_prices)
close_low = (close_prices-low_prices) < 0.1*(high_prices - low_prices)
big_bounce = (close_prices - low_prices) > 0.15
rng = (high_prices - low_prices)/close_prices > 0.075

I =  big_downday.shift(3) & high_volume.shift(3) & (ret_5d.shift(3) < 0) & (ret.shift(0)<0) #& (ret.shift(-1)<0)
I = (ret_20d>0.1) & close_low & rng

return_fwd = (close_prices.shift(-2)/close_prices.shift(0)-1)
returns = return_fwd.copy()

#return_fwd_2d = (close_prices.shift(-2)/close_prices.shift(-1)-1)
returns_strategy = I*returns

print("Strategy mean return per day")
print(returns_strategy[returns_strategy!=0].stack().mean())
print("Strategy daily volatility")
print(returns_strategy[returns_strategy!=0].stack().std())
kelly= returns_strategy[returns_strategy!=0].stack().mean()/(returns_strategy[returns_strategy!=0].stack().std()**2)
print("Strategy Kelly fraction")
print(kelly)

#calculate returns
long_ret = returns_strategy.fillna(0)
long_cum_ret = np.cumprod(long_ret+1)
#short_ret = -1*short_pos*(mod_opcl_returns+0.03/100)
#short_cum_ret = np.cumprod(short_ret+1)
strat_ret = long_ret[long_ret!=0].mean(axis=1).fillna(0) #+short_ret
cum_ret = np.cumprod(strat_ret+1)
plt.figure()
plt.plot(cum_ret.ffill())


liquidated_stocks = big_downday.shift(2) & high_volume.shift(2) & (ret_5d.shift(2) < 0)
#list the values for change in liquidity by name
liquidated_stocks_last = liquidated_stocks.tail(1).T

#remove all names with z-score less than 2
# I_names = I_last[I_last[I]]
# I_last.dropna(how='any',inplace=True)

#I_last.drop(I_last[I_last,inplace=True)

names = liquidated_stocks_last.set_axis(['Liquidated names'], axis=1)
names_sorted = names.sort_values(by=['Liquidated names'])

output_liquidated_stocks = pd.DataFrame()
output_liquidated_stocks["Names"] = names_sorted