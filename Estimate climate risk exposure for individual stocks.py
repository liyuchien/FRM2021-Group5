#!/usr/bin/env python
# coding: utf-8

# # TESTING

# In[ ]:


import pandas_datareader as reader
import pandas as pd
import datetime as dt
import statsmodels.api as sm
import os

end = dt.date(2014,12,31)
start = dt.date(2014,1,1)
stocks = ['AAPL']
stocksret = reader.get_data_yahoo(stocks,start,end)['Adj Close'].pct_change()
stocksret_month = stocksret.resample('M').agg(lambda x:(x+1).prod()-1)
stocksret_month = stocksret_month[1:]
factors = reader.DataReader('F-F_Research_Data_Factors','famafrench',start,end)[0]
stocksret_month.index = factors.index
df = pd.merge(stocksret_month,factors,on='Date')
df[['Mkt-RF','SMB','HML','RF']]/100
df['Ri-RF'] = df.AAPL-df.RF
print(df)
# regression
y = df['Ri-RF']
x = df[['Mkt-RF','SMB','HML']]
x_sm = sm.add_constant(x)
model = sm.OLS(y,x_sm)
results = model.fit()
results.summary()


# # 1

# In[ ]:


#date
end = dt.date(2016,12,31)
start = dt.date(2014,1,1)
#Nasdaq traded symbol
stocks = reader.nasdaq_trader.get_nasdaq_symbols(retry_count=3, timeout=30, pause=None)['NASDAQ Symbol']
stocks.to_excel(r'C:\Users\USER\FRM_Python\Nasdaq_list.xlsx',index=True)
#yahho finance daily return
stocksret = reader.get_data_yahoo(stocktest,start,end)['Adj Close'].pct_change()
stocksret = stocksret[1:]
stocksret = pd.DataFrame(stocksret)
stocksret.columns = stocks
stocksret
#save as xlsx or csv
stocksret.to_excel(r'C:\Users\USER\FRM_Python\Nasdaq_daily.xlsx',index=True)

### Or translate to monthly return as paper does ###
#stocksret_month = stocksret.resample('M').agg(lambda x:(x+1).prod()-1)
#stocksret_month = stocksret_month[1:]
#stocksret_month = pd.DataFrame(stocksret_month)
#stocksret_month.columns = stocks
#df = stocksret_month
#df= df.drop([col for col in df.columns if df[col].eq(0).any()], axis=1)
#df.to_excel(r'C:\Users\USER\FRM_Python\Nasdaq.xlsx',index=True)


# In[ ]:


#補齊資料


# In[ ]:


#補齊遺漏stocks
a = pd.read_excel('Nasdaq_daily.xlsx')
mylist = a.columns
mylist = pd.Series(mylist)
mylist= mylist[1:]
b= pd.read_excel('Nasdaq_list.xlsx')
Naslist = b['NASDAQ_Symbol']
Naslist = pd.Series(Naslist)
x = pd.concat([mylist,Naslist])
x_trim = x.drop_duplicates(keep=False, inplace=False)
missing = pd.DataFrame(x_trim)
missing.to_excel(r'C:\Users\USER\FRM_Python\missing.xlsx',index=True)
#run again and merge them
df = pd.merge(a,missing,on='Date')
df.to_excel(r'C:\Users\USER\FRM_Python\Nasdaq_all_dropempty.xlsx',index=True)


# In[ ]:


#Fama-French 3 factors


# In[ ]:


#df.set_index('Date')
mfactors = reader.DataReader('F-F_Research_Data_Factors_Daily','famafrench',start,end)[0]
df.rename_axis("Symb", axis="columns")
df.set_index('Date')
df.drop(['Unnamed: 0'], axis=1)
k = df.drop(axis=1,columns='Date')
k.index = mfactors.index
df = pd.merge(k,mfactors,on='Date')
df.drop(['Unnamed: 0'], axis=1)
df['Mkt-RF']= df['Mkt-RF']/100
df['SMB_y']= df['SMB_y']/100
df['HML']= df['HML']/100
df['RF_y']= df['RF_y']/100
dft = df


# In[ ]:


#Merge data on date


# In[ ]:


NAS = dft.columns[:-4]
for i in NAS:
    dft[i+'-Rf'] = dft[i]-df['RF_y']
dfy = dft[dft.columns[4024:]]
# change name
dfy = dfy.rename(columns={"RF_y": "RF"})
dfy = dfy.rename(columns={"SMB_y": "SMB"})
dfy = dfy.rename(columns={"Mkt-RF": "RMRF"})
dfy = dfy.drop(axis=1,columns='Unnamed: 0-Rf')
dfy = dfy.reset_index()
for i in range(len(dfy['Date'])):
    dfy['Date'][i] =str(dfy['Date'][i]).split(' ')[0]
dfy = dfy[:607]
cc = pd.read_excel('daily_residual.xlsx')
cc.columns = ['Date', 'CC']
for i in range(len(cc['Date'])):
    cc['Date'][i] = cc['Date'][i].replace('/', '-')
cc['Date']
merge = cc.merge(dfy, how='inner', on='Date')
merge1 = merge[merge.columns[6:]]
mergr1.to_excel(r'C:\Users\USER\FRM_Python\    .xlsx',index=True)


# In[ ]:


#Regression 1


# In[ ]:


#除理資料結束；Regression 1 開始。
#建立一個空dataframe
dfempty = pd.DataFrame(index=['const','RMRF','SMB','HML','CC'],columns=merge.columns[5:])
dfe = dfe.drop(axis=1,columns='RF')
for i in dfe.columns:
    y = merge1[i]
    x = dfe[i]
    model = sm.OLS(y,x_sm)
    results = model.fit()
    dfempty.at['B_RMRF',i] = results.params.RMRF
    dfempty.at['B_SMB', i] = results.params.SMB
    dfempty.at['B_HML', i] = results.params.HML
    dfempty.at['B_CC', i] = results.params.CC
dfempty.to_excel('Reggression1.xlsx')


# In[ ]:


#Regression 2


# In[ ]:


dfempty2 = pd.DataFrame(index=['BetaRMRFG','BetaSMBG','BetaHMLG','Betacc'],columns=merge1.columns)
#Fama_daily_regression.xlsx excel是我們用excel整理前面的資料
fin = pd.read_excel('Fama_daily_regression.xlsx')
fin.columns[2:6]
for i in fin.columns[11:]:
    y = fin[i]
    x = fin[['HMLG', 'SMBG', 'RMRFG', 'Cfactor']]
    model = sm.OLS(y,x)
    results = model.fit()
    dfempty2.at['BetaRMRFG', i] = results.params.RMRFG
    dfempty2.at['BetaSMBG', i] = results.params.SMBG
    dfempty2.at['BetaHMLG', i] = results.params.HMLG
    dfempty2.at['Betacc', i] = results.params.Cfactor
dfempty2.to_excel('Final.xlsx')

