# -*- coding: utf-8 -*-
"""
Created on Tue Dec  7 15:19:58 2021

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

re_names = "EVO.ST "\
"INVE-B.ST "\
"HM-B.ST "\
"ICA.ST "\
"ERIC-B.ST "\
"NDA-SE.ST "\
"AZN.ST "\
"VOLV-B.ST "\
"GETI-B.ST "\
"SAND.ST "\
"SWMA.ST "\
"ATCO-A.ST "\
"SWED-A.ST "\
"BOL.ST "\
"ESSITY-B.ST "\
"SEB-A.ST "\
"EMBRAC-B.ST "\
"SINCH.ST "\
"ASSA-B.ST "\
"HEXA-B.ST "\
"LUNE.ST "\
"ABB.ST "\
"NIBE-B.ST "\
"EQT.ST "\
"SKF-B.ST "\
"VOLCAR-B.ST "\
"ELUX-B.ST "\
"TELIA.ST "\
"SBB-B.ST "\
"SHB-A.ST "\
"LIFCO-B.ST "\
"SAGA-B.ST "\
"TEL2-B.ST "\
"ALFA.ST "\
"ATCO-B.ST "\
"CAST.ST "\
"ONCO.ST "\
"SSAB-B.ST "\
"SCA-B.ST "\
"ALIV-SDB.ST "\
"KINV-B.ST "\
"EPI-A.ST "\
"HUSQ-B.ST "\
"SKA-B.ST "\
"SF.ST "\
"BALD-B.ST "\
"SOBI.ST "\
"TIGO-SDB.ST "\
"NIVI-B.ST "\
"HTRO.ST "\
"TREL-B.ST "\
"SECU-B.ST "\
"INDU-C.ST "\
"EKTA-B.ST "\
"KIND-SDB.ST "\
"SSAB-A.ST "\
"VNE-SDB.ST "\
"AMAST.ST "\
"LUND-B.ST "\
"AZA.ST "\
"LATO-B.ST "\
"SAVE.ST "\
"EPI-B.ST "\
"STOR-B.ST "\
"AXFO.ST "\
"PCELL.ST "\
"KLARA-B.ST "\
"SHOT.ST "\
"DOM.ST "\
"SECT-B.ST "\
"INDT.ST "\
"BICO.ST "\
"ADDT-B.ST "\
"BILL.ST "\
"CTEK.ST "\
"TRUE-B.ST "\
"THULE.ST "\
"FABG.ST "\
"HOLM-B.ST "\
"HUFV-A.ST "\
"LOGI-A.ST "\
"ALIF-B.ST "\
"INTRUM.ST "\
"KAMBI.ST "\
"SAS.ST "\
"MIPS.ST "\
"FING-B.ST "\
"VESTUM.ST "\
"AAK.ST "\
"VITR.ST "\
"KDEV.ST "\
"WIHL.ST "\
"HUMBLE.ST "\
"NENT-B.ST "\
"BHG.ST "\
"JM.ST "\
"AEGIR.ST "\
"INSTAL.ST "\
"BEIJ-B.ST "\
"SAAB-B.ST "\
"HPOL-B.ST "\
"CTM.ST "\
"STORY-B.ST "\
"BOOZT.ST "\
"CIBUS.ST "\
"LIAB.ST "\
"NYF.ST "\
"VNV.ST "\
"WALL-B.ST "\
"STE-R.ST "\
"CINT.ST "\
"BETS-B.ST "\
"SWEC-B.ST "\
"ALLR.ST "\
"NETI-B.ST "\
"HEM.ST "\
"VIT-B.ST "\
"VIMIAN.ST "\
"BURE.ST "\
"RATO-B.ST "\
"DIOS.ST "\
"PNDX-B.ST "\
"LOOMIS.ST "\
"AFRY.ST "\
"LUMI.ST "\
"SUS.ST "\
"PDX.ST "\
"NEWA-B.ST "\
"NCC-B.ST "\
"SECARE.ST "\
"ARJO-B.ST "\
"ATRLJ-B.ST "\
"CORE-B.ST "\
"COOR.ST "\
"SFAB.ST "\
"MEKO.ST "\
"BMAX.ST "\
"SDIP-B.ST "\
"IPCO.ST "\
"CATE.ST "\
"CALTX.ST "\
"TROAX.ST "\
"LEO.ST "\
"NOLA-B.ST "\
"LOGI-B.ST "\
"BUFAB.ST "\
"LAGR-B.ST "\
"SKIS-B.ST "\
"MTG-B.ST "\
"PEAB-B.ST "\
"MYCR.ST "\
"CLAS-B.ST "\
"SYNSAM.ST "\
"AOI.ST "\
"HMS.ST "\
"ANOD-B.ST "\
"BRAV.ST "\
"SVOL-B.ST "\
"BILI-A.ST "\
"BIOT.ST "\
"TOBII.ST "\
"RESURS.ST "\
"SCST.ST "\
"G5EN.ST "\
"M8G.ST "\
"RVRC.ST "\
"BONAV-B.ST "\
"GRNG.ST "\
"NOBI.ST "\
"BETCO.ST "\
"NOBINA.ST "\
"NCAB.ST "\
"VOLO.ST "\
"JOMA.ST "\
"INWI.ST "\
"EOLU-B.ST "\
"MTRS.ST "\
"HNSA.ST "\
"CRED-A.ST "\
"EXS.ST "\
"NOTE.ST "\
"NP3.ST "\
"GARO.ST "\
"BIOA-B.ST "\
"RENEW.ST "\
"FNM.ST "\
"OX2.ST "\
"CLA-B.ST "\
"SEYE.ST "\
"MCOV-B.ST "\
"EPRO-B.ST "\
"DSNO.ST "\
"COIC.ST "\
"COALA.ST "\
"ENQ.ST "\
"DUST.ST "\
"BALCO.ST "\
"TRANS.ST "\
"AMBEA.ST "\
"COLL.ST "\
"KNOW.ST "\
"TETY.ST "\
"SOLT.ST "\
"CAMX.ST "\
"CI-B.ST "\
"ATT.ST "\
"AAC.ST "\
"ALIG.ST "\
"8TRA.ST "\
"CARY.ST "\
"ACAD.ST "\
"GENO.ST "\
"CANTA.ST "\
"BEIA-B.ST "\
"SIGNUP.ST "\
"BFG.ST "\
"FG.ST "\
"IVACC.ST "\
"ISOFOL.ST "\
"EG7.ST "\
"SEDANA.ST "\
"ACCON.ST "\
"ISR.ST "\
"NWG.ST "\
"ACAST.ST "\
"XVIVO.ST "\
"OEM-B.ST "\
"BONEX.ST "\
"SIVE.ST "\
"THUNDR.ST "\
"EXPRS2.ST "\
"AZELIO.ST "\
"CEVI.ST "\
"HANZA.ST "\
"BERG-B.ST "\
"TFBANK.ST "\
"ASAB.ST "\
"BRG-B.ST "\
"BULTEN.ST "\
"HLDX.ST "\
"BIOG-B.ST "\
"BUSER.ST "\
"KFAST-B.ST "\
"IMP-A-SDB.ST "\
"INTEG-B.ST "\
"FLAT-B.ST "\
"KAR.ST "\
"IMMNOV.ST "\
"DOXA.ST "\
"LUG.ST "\
"CS.ST "\
"GREEN.ST "\
"READ.ST "\
"NITRO.ST "\
"PRIC-B.ST "\
"PLEX.ST "\
"MAHA-A.ST "\
"HUM.ST "\
"PIEZO.ST "\
"BTS-B.ST "\
"MANTEX.ST "\
"FASTAT.ST "\
"OASM.ST "\
"OPTI.ST "\
"IRIS.ST "\
"STEF-B.ST "\
"FPIP.ST "\
"VEFAB.ST "\
"CAT-B.ST "\
"FAG.ST "\
"CDON.ST "\
"LINC.ST "\
"MCAP.ST "\
"HOFI.ST "\
"RAY-B.ST "\
"EAST.ST "\
"DUNI.ST "\
"IDUN-B.ST "\
"PLAZ-B.ST "\
"BEGR.ST "\
"AWRD.ST"

#=============================================================================
# ============================================================================
hist = yf.download(re_names, start='2015-01-01', end=tday_str)
# ============================================================================
#=================================================================



close_prices = hist["Adj Close"]#.dropna(how='all').fillna(0)
volumes = hist["Volume"].dropna(how='all').fillna(0)

r_vol=volumes/volumes.rolling(100).mean().shift(1)


#calculate daily returns
ret_daily = close_prices.pct_change()

#calculate 5 day returns
ret_5d = close_prices.pct_change(5)
# =============================================================================
# 
# #generate position indicator bottom 20% = +1 top 25% = -1, exclude stocks with short sale restrictions from top 20%
# #short_sale_restrict = ["FPAR-A.ST", "KFAST-B.ST", "DIOS.ST", "HEBA-B.ST", "TRIAN-B.ST", "CIBUS.ST", "AMAST.ST"]
# #ret_5d_shortable = ret_5d.drop(short_sale_restrict, axis=1)
# #ret_daily_shortable = ret_daily.drop(short_sale_restrict, axis=1)
# percentile80_shortable = ret_5d.quantile(0.9,axis=1)
# short_ind = ret_5d.ge(percentile80_shortable,axis=0)
# #replace false with NaN to get the right average
# short_ind = short_ind.replace(False, np.nan)
# short_returns_daily = -ret_daily*short_ind.shift(1)
# =============================================================================





#long book position indicator
percentile20 = ret_5d.quantile(0.1,axis=1)


#create binary dataframe to exclude stocks with big move large volume days in the last 3 sessions
significant_days = (r_vol > 5) & (ret_daily < -0)

not_excluded = significant_days.rolling(3).sum() < 1


#calculate average daily turnover
ADT = close_prices.rolling(20).mean().shift(1)*volumes.rolling(20).mean().shift(1)/1000000


long_ind = ret_5d.le(percentile20,axis=0) & not_excluded & (ADT > 3)
#replace false with NaN to avoid 0s impacting the mean
long_ind = long_ind.replace(False, np.nan)
long_returns_daily = ret_daily*long_ind.shift(1)

long_returns_daily.mean(axis=0)
n_longs = long_ind.count(axis=1)

#calc transaction cost
trans = long_ind-long_ind.shift(1)
n_trans = trans.count().sum()

trans_value = n_trans*100000
total_trans_cost = n_trans*29

trans_proc_fee = total_trans_cost/trans_value

#daily returns of long short strategy
#avg_long_ret = starting_capital*long_returns_daily.mean(axis=1)-transaction_cost
avg_long_ret = long_returns_daily.mean(axis=1)-trans_proc_fee
#avg_short_ret = short_returns_daily.mean(axis=1)-trans_proc_fee
daily_returns_strat = avg_long_ret #+avg_short_ret

#avg_daily_rets  = daily_returns_strat.mean(axis=1)

#Cumulative returns 
#cum_ret =starting_capital +  np.cumsum(daily_returns_strat) #
cum_ret =(1 + daily_returns_strat).cumprod()
#cum_long_ret =  (1 + avg_long_ret).cumprod()
#cum_short_ret =  (1 + avg_short_ret).cumprod()


###########################################
#stats for basic strategy
##########################################

print("   ")
print('Short term reversal INDUSTRIALS')
mean_ret = cum_ret.tail(1)**(1/7)-1
print("CAGR " + str(mean_ret[0]))
vol = (daily_returns_strat.std()*math.sqrt(252))
sharpe = mean_ret/vol
kelly_f = mean_ret/vol**2
print("Volatility " + str(vol))
print("Sharpe " + str(sharpe[0]))
print("Kelly fraction " + str(kelly_f[0]))
#maxiumum drawdown
Roll_Max = cum_ret.cummax()
Daily_Drawdown = cum_ret/Roll_Max - 1.0
Max_Daily_Drawdown = Daily_Drawdown.cummin()
print("Max drawdown " + str(Max_Daily_Drawdown.tail(1)[0]))

#plots
plt.plot(cum_ret)

###################################################
#modified strategy considering factor momentum
####################################################

mom_cum_ret = (1+daily_returns_strat[cum_ret.pct_change(20).shift(1) > 0]).cumprod()
#mom_cum_ret = starting_capital + np.cumsum(daily_returns_strat[cum_ret.pct_change(20).shift(1) > 0])
mom_daily_ret_IND = mom_cum_ret.pct_change()


mom_mean_ret = mom_cum_ret.tail(1)**(1/7)-1

mom_vol = (daily_returns_strat[cum_ret.pct_change(40).shift(1) > 0].std()*math.sqrt(252))
mom_sharpe = mom_mean_ret/mom_vol
mom_kelly_f = mom_mean_ret/mom_vol**2

#maxiumum drawdown
mom_Roll_Max = mom_cum_ret.cummax()
mom_Daily_Drawdown = mom_cum_ret/mom_Roll_Max - 1.0
mom_Max_Daily_Drawdown = mom_Daily_Drawdown.cummin()
print("   ")
print('Short term reversal with factor momentum INDUSTRIALS')
print("CAGR " + str(mom_mean_ret[0]))
print("Volatility " + str(mom_vol))

print("Sharpe " + str(mom_sharpe[0]))
print("Kelly fraction " + str(mom_kelly_f[0]))
#maxiumum drawdown
Roll_Max = cum_ret.cummax()
Daily_Drawdown = cum_ret/Roll_Max - 1.0
Max_Daily_Drawdown = Daily_Drawdown.cummin()
print("Max drawdown " + str(mom_Max_Daily_Drawdown.tail(1)[0]))

#calculate log returns st reversal momentum strategy and print returns per year
mom_log_ret_IND = np.log(mom_cum_ret)-np.log(mom_cum_ret.shift(1))
per = mom_log_ret_IND.index.to_period("Y")
g = mom_log_ret_IND.groupby(per)
ret_per_year = g.sum()
print("   ")
print("st reversal INDUSTRIALS with factor momentum returns per year")
print(ret_per_year)


per_M = mom_log_ret_IND.index.to_period("M")
grouping_month = mom_log_ret_IND.groupby(per_M)
ret_per_month = grouping_month.sum()
#stats for monthly returns
percent_positive = ret_per_month[ret_per_month>0].count()/ret_per_month.count()
print("")
print("percent positive months " + str(percent_positive))


plt.plot(mom_cum_ret)


################
#buy and hold
################
avg_ret_boh= ret_daily.mean(axis=1)
cum_ret_boh =  (1 + avg_ret_boh).cumprod()
#avg_ret_boh= starting_capital*ret_daily.mean(axis=1)
#cum_ret_boh =  starting_capital + np.cumsum(avg_ret_boh)
plt.plot(cum_ret_boh)


#stats buy and hold
print("   ")
print('Buy and hold stats')
boh_mean_ret = cum_ret_boh.tail(1)**(1/7)-1
boh_vol = (avg_ret_boh.std()*math.sqrt(252))
boh_sharpe = boh_mean_ret/boh_vol
boh_kelly_f = boh_mean_ret/boh_vol**2

#maxiumum drawdown
boh_Roll_Max = cum_ret_boh.cummax()
boh_Daily_Drawdown = cum_ret_boh/boh_Roll_Max - 1.0
boh_Max_Daily_Drawdown = boh_Daily_Drawdown.cummin()



print("CAGR " + str(boh_mean_ret[0]))
print("Volatility " + str(boh_vol))

print("Sharpe " + str(boh_sharpe[0]))
print("Kelly fraction " + str(boh_kelly_f[0]))

print("Max drawdown " + str(boh_Max_Daily_Drawdown.tail(1)[0]))

print(" ")

print('40-day momentum of short term reversal INDUSTRIALS strategy')
print(cum_ret.pct_change(40).tail(1))

