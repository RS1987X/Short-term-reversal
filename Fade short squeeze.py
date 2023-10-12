# -*- coding: utf-8 -*-
"""
Created on Wed Oct 11 08:17:48 2023

@author: Richard
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


data = pd.read_csv(r".\universe all listed sweden.csv",delimiter=";")

yf_names = data["Yahoo finance name"]

tickers = yf_names.str.cat(sep=" ")


# =============================================================================
# ============================================================================
hist = yf.download(tickers, start='2015-01-01', end=tday_str)
# ============================================================================
# =============================================================================

# .dropna(how='all',inplace = True)#.fillna(0)
close_prices = hist["Adj Close"]
#close_prices.dropna(axis=0, how='all', inplace=True)
close_prices_ffill = close_prices.ffill(axis=0)

open_prices = hist["Open"]
#open_prices.dropna(axis=0, how='all', inplace=True)
open_prices_ffill = open_prices.ffill(axis=0)
    
#close_prices = close_prices.drop([pd.Timestamp('2023-05-31 00:00:00')])#, pd.Timestamp(
 #   '2018-06-22 00:00:00'), pd.Timestamp('2017-06-23 00:00:00'), pd.Timestamp('2017-06-06 00:00:00')])
high_prices = hist["High"]
high_prices_ffill = high_prices.ffill(axis=0)
low_prices = hist["Low"]
low_prices_ffill = low_prices.ffill(axis=0)
#high_prices = high_prices.drop([pd.Timestamp('2018-06-06 00:00:00'), pd.Timestamp(
#    '2018-06-22 00:00:00'), pd.Timestamp('2017-06-23 00:00:00'), pd.Timestamp('2017-06-06 00:00:00')])

volumes = hist["Volume"]#.dropna(how='all')  # .fillna(0)
#volumes.dropna(axis=0, how='all', inplace=True)
volumes_ffill = volumes.ffill(axis=0)
#volumes = volumes.drop([pd.Timestamp('2018-06-06 00:00:00'), pd.Timestamp('2018-06-22 00:00:00'),
#                       pd.Timestamp('2017-06-23 00:00:00'), pd.Timestamp('2017-06-06 00:00:00')])

avg_volume = volumes_ffill.shift(0).rolling(10).mean()
ret = close_prices_ffill/close_prices_ffill.shift(1)-1

avg_price = (close_prices_ffill + open_prices_ffill +high_prices_ffill + low_prices_ffill)/4

turnover = volumes_ffill*(close_prices_ffill + open_prices_ffill +high_prices_ffill + low_prices_ffill)/4
avg_turnover = turnover.shift(0).rolling(10).mean()


amihud_il = 1000000*(abs(ret.shift(1))/turnover.shift(1))

avg_amihud_il = amihud_il.rolling(10).mean().ffill(axis=0)


liq_threshold_1 = avg_amihud_il.quantile(q=0.2,axis=1)
liq_threshold_2 = avg_amihud_il.quantile(q=0.4,axis=1)
liq_threshold_3 = avg_amihud_il.quantile(q=0.6,axis=1)
liq_threshold_4 = avg_amihud_il.quantile(q=0.8,axis=1)


liq_segment_1 = avg_amihud_il.le(liq_threshold_1,axis=0)
liq_segment_2 = avg_amihud_il.ge(liq_threshold_1,axis=0) & avg_amihud_il.le(liq_threshold_2,axis=0)
liq_segment_3 = avg_amihud_il.ge(liq_threshold_2,axis=0) & avg_amihud_il.le(liq_threshold_3,axis=0)
liq_segment_4 = avg_amihud_il.ge(liq_threshold_3,axis=0) & avg_amihud_il.le(liq_threshold_4,axis=0)
liq_segment_5 = avg_amihud_il.ge(liq_threshold_4,axis=0)


A = abs(high_prices_ffill-close_prices_ffill.shift(1))
B = abs(low_prices_ffill-close_prices_ffill.shift(1))
C = abs(high_prices_ffill - low_prices_ffill)
TR = pd.concat([A,B,C]).max(level=0)
ATR = TR.shift(1).rolling(20).mean()


ret_3d = close_prices_ffill.shift(1)/close_prices_ffill.shift(4)-1
ret_5d = close_prices_ffill.shift(1)/close_prices_ffill.shift(6)-1
ret_20d = close_prices_ffill.shift(1)/close_prices_ffill.shift(21)-1
close_low = (close_prices-low_prices) < 0.2*(high_prices - low_prices)
close_high = (high_prices - close_prices) < 0.1*(high_prices - low_prices)
consecutive_up_days = (ret.shift(1) > 0) & (ret.shift(2) > 0) & (ret.shift(3) > 0) #& (ret.shift(4) > 0)

ret_40d = close_prices_ffill.shift(1)/close_prices_ffill.shift(41)-1

q = ret_40d.quantile(0.1,axis=1)
loser = ret_40d.le(q,axis=0)

######################
######################

r_vol = volumes_ffill/avg_volume

ret_yest = close_prices_ffill.shift(1)/close_prices_ffill.shift(2)-1

two_day_high = high_prices_ffill.shift(1).rolling(2).max()

# & (ret_yest.shift(1) < 0) &
I = (ret_yest < 0) & (ret_yest.shift(1) < 0) & (close_prices_ffill > two_day_high) & (ret_20d < -0.1) & liq_segment_1


#holding_period = 1
#no I in last equivalent holding period
#I_mod =  (I.shift(1).rolling(holding_period).sum() < 1) & I

fill = close_prices_ffill

returns = (fill-close_prices_ffill.shift(-1))/fill

#returns_copy() = returns.copy()
returns_strategy =  I*returns

mean = returns_strategy[returns_strategy!=0].stack().mean()
vol = returns_strategy[returns_strategy!=0].stack().std()

print("Strategy mean return per day")
print(mean)
print("Strategy daily volatility")
print(vol)
kelly= returns_strategy[returns_strategy!=0].stack().mean()/(returns_strategy[returns_strategy!=0].stack().std()**2)
print("Strategy Kelly fraction")
print(kelly)
print("Trimmed mean")
print(stats.trim_mean(returns_strategy[returns_strategy!=0].stack(),0.05))

#calculate returns
long_ret = returns_strategy.fillna(0)
long_cum_ret = np.cumprod(long_ret+1)
#short_ret = -1*short_pos*(mod_opcl_returns+0.03/100)
#short_cum_ret = np.cumprod(short_ret+1)
strat_ret = long_ret[long_ret!=0].mean(axis=1).fillna(0) #+short_ret
cum_ret = np.cumprod(strat_ret+1)
plt.figure()
plt.plot(cum_ret.ffill())


rolling_high = cum_ret.cummax()
draw_down = cum_ret/rolling_high-1
max_dd = draw_down.cummin().tail(1)
print("Max draw down")
print(max_dd)

#calculate log returns by year
log_cum_ret = np.log(strat_ret+1)
per = log_cum_ret.index.to_period("Y")
g = log_cum_ret.groupby(per)
ret_per_year = g.sum()
print("   ")
print("Gap down long returns by year")
print(ret_per_year)

#calculate log returns by month
month_per = log_cum_ret.index.to_period("M")
returns_by_month = log_cum_ret.groupby(log_cum_ret.index.month).sum()
print("   ")
print("Gap down long returns by month")
print(returns_by_month)

#calculate volatility by month
month_per = log_cum_ret.index.to_period("M")
vol_by_month = log_cum_ret.groupby(log_cum_ret.index.month).std()
print("   ")
print("Continuation break out volatility by month")
print(vol_by_month)


