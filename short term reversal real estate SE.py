# -*- coding: utf-8 -*-
"""
Created on Tue Nov 23 09:55:31 2021

@author: Richard
"""



import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
import math
from datetime import date

#get prices from yahoo finance

tday = date.today()
tday_str = tday.strftime("%Y-%m-%d")
#=============================================================================
# =============================================================================
hist = yf.download('BALD-B.ST BRIN-B.ST SAGA-B.ST SBB-B.ST CAST.ST WALL-B.ST FABG.ST CORE-B.ST WIHL.ST'
                    ' NYF.ST HUFV-A.ST KLED.ST FPAR-A.ST ATRLJ-B.ST CATE.ST NP3.ST PLAZ-B.ST'
                    ' KFAST-B.ST DIOS.ST HEBA-B.ST TRIAN-B.ST CIBUS.ST BONAV-B.ST AMAST.ST'
                    ' PNDX-B.ST JM.ST', start='2015-01-01', end=tday_str)
# =============================================================================
#=============================================================================

close_prices = hist["Adj Close"]#.dropna(how='all').fillna(0)
volumes = hist["Volume"].dropna(how='all').fillna(0)
close_prices = close_prices.drop('2020-01-01')

#calculate daily returns
ret_daily = close_prices.pct_change()

#calculate 5 day returns
ret_5d = close_prices.pct_change(5)

#generate position indicator bottom 20% = +1 top 25% = -1, exclude stocks with short sale restrictions from top 20%
short_sale_restrict = ["FPAR-A.ST", "KFAST-B.ST", "DIOS.ST", "HEBA-B.ST", "TRIAN-B.ST", "CIBUS.ST", "AMAST.ST"]
ret_5d_shortable = ret_5d.drop(short_sale_restrict, axis=1)
ret_daily_shortable = ret_daily.drop(short_sale_restrict, axis=1)
percentile80_shortable = ret_5d_shortable.quantile(0.75,axis=1)
short_ind = ret_5d_shortable.ge(percentile80_shortable,axis=0)
#replace false with NaN to get the right average
short_ind = short_ind.replace(False, np.nan)
short_returns_daily = -ret_daily_shortable*short_ind.shift(1)


percentile20 = ret_5d.quantile(0.2,axis=1)
long_ind = ret_5d.le(percentile20,axis=0)
#replace false with NaN to avoid 0s impacting the mean
long_ind = long_ind.replace(False, np.nan)
long_returns_daily = ret_daily*long_ind.shift(1)


n_longs = long_ind.count(axis=1)
transaction_cost = 0.0004#n_longs*29#0.00058
starting_capital = 500000

#daily returns of long short strategy
#avg_long_ret = starting_capital*long_returns_daily.mean(axis=1)-transaction_cost
avg_long_ret = long_returns_daily.mean(axis=1)-transaction_cost
avg_short_ret = short_returns_daily.mean(axis=1)-transaction_cost
daily_returns_strat = avg_long_ret #+avg_short_ret

#avg_daily_rets  = daily_returns_strat.mean(axis=1)

#Cumulative returns 
#cum_ret =starting_capital +  np.cumsum(daily_returns_strat) #
cum_ret =(1 + daily_returns_strat).cumprod()
#cum_long_ret =  (1 + avg_long_ret).cumprod()
#cum_short_ret =  (1 + avg_short_ret).cumprod()

#stats
print('Short term reversal')
mean_ret = cum_ret.tail(1)**(1/7)-1
print(mean_ret)
vol = (daily_returns_strat.std()*math.sqrt(252))
sharpe = mean_ret/vol
kelly_f = mean_ret/vol**2
print(vol)
print(sharpe)
print(kelly_f)
#maxiumum drawdown
Roll_Max = cum_ret.cummax()
Daily_Drawdown = cum_ret/Roll_Max - 1.0
Max_Daily_Drawdown = Daily_Drawdown.cummin()
print(Max_Daily_Drawdown.tail(1))

#plos
plt.plot(cum_ret)
#plt.plot(cum_long_ret)
#plt.plot(cum_short_ret)
#plt.plot(Daily_Drawdown)

#consider factor momentum
mom_cum_ret = (1+daily_returns_strat[cum_ret.pct_change(20).shift(1) > 0]).cumprod()
#mom_cum_ret = starting_capital + np.cumsum(daily_returns_strat[cum_ret.pct_change(20).shift(1) > 0])
mom_daily_ret_RE = mom_cum_ret.pct_change()


mom_mean_ret = mom_cum_ret.tail(1)**(1/7)-1
print('Short term reversal with factor momentum')
print(mom_mean_ret)
mom_vol = (daily_returns_strat[cum_ret.pct_change(20).shift(1) > 0].std()*math.sqrt(252))
mom_sharpe = mom_mean_ret/mom_vol
mom_kelly_f = mom_mean_ret/mom_vol**2
print(mom_vol)
print(mom_sharpe)
print(mom_kelly_f)
#maxiumum drawdown
mom_Roll_Max = mom_cum_ret.cummax()
mom_Daily_Drawdown = mom_cum_ret/mom_Roll_Max - 1.0
mom_Max_Daily_Drawdown = mom_Daily_Drawdown.cummin()
print(mom_Max_Daily_Drawdown.tail(1))

    
plt.plot(mom_cum_ret)

#buy and hold
avg_ret_boh= ret_daily.mean(axis=1)
cum_ret_boh =  (1 + avg_ret_boh).cumprod()
#avg_ret_boh= starting_capital*ret_daily.mean(axis=1)
#cum_ret_boh =  starting_capital + np.cumsum(avg_ret_boh)
plt.plot(cum_ret_boh)


#stats buy and hold
print('Buy and hold stats')
boh_mean_ret = cum_ret_boh.tail(1)**(1/7)-1
print(boh_mean_ret)
boh_vol = (avg_ret_boh.std()*math.sqrt(252))
boh_sharpe = boh_mean_ret/boh_vol
boh_kelly_f = boh_mean_ret/boh_vol**2
print(boh_vol)
print(boh_sharpe)
print(boh_kelly_f)
#maxiumum drawdown
boh_Roll_Max = cum_ret_boh.cummax()
boh_Daily_Drawdown = cum_ret_boh/boh_Roll_Max - 1.0
boh_Max_Daily_Drawdown = boh_Daily_Drawdown.cummin()
print(boh_Max_Daily_Drawdown.tail(1))

print('20-day momentum of short term reversal REAL ESTATE strategy')
print(cum_ret.pct_change(20).tail(1))


