#!/usr/bin/env python
# coding: utf-8

# # Import Package dan Data Collecting

# In[1]:


# pip install pmdarima


# In[1]:


#======= Modelling =====
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from statsmodels.tsa.stattools import adfuller
from pmdarima import auto_arima
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from pmdarima.arima.utils import ndiffs

#======= Pandas confiq =====
import pandas as pd

#==== Numpy ======
import numpy as np
from math import sqrt

#===== Visualization =====
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_theme(style="darkgrid")

import warnings
warnings.filterwarnings('ignore')


# In[2]:


customer = pd.read_csv('Case Study - Customer.csv', sep=';')
product = pd.read_csv('Case Study - Product.csv', sep=';')
store = pd.read_csv('Case Study - Store.csv', sep=';')
transaction = pd.read_csv('Case Study - Transaction.csv', sep=';')


# # Data Preprocessing

# In[3]:


transaction = transaction.astype({
    'Price':'float',
    'TotalAmount':'float',
    'Date':'datetime64'
})

transaction.info()


# In[4]:


product = product.astype({
    'price':'float'
})

product.info()


# In[5]:


df = transaction.merge(product,how='left', left_on='ProductID', right_on='productid')
df.head()


# In[6]:


df_reg = df.groupby(['Date']).agg({'Qty':'sum'})
df_reg


# # Exploratory Data Analysis

# In[7]:


sns.boxplot(x=df_reg['Qty'])


# In[8]:


df_reg.plot()


# Dari visualisasi diatas, data Qty tidak terdapat pola trend naik maupun turun dan tidak terdapat juga seasonality pada Qty. Dat Qty cederung memiliki pola tetap atau stasioner

# **CEK STATIONARY**

# In[9]:


def ad_test(data):
    dftest = adfuller(data, autolag='AIC')
    print('1. ADF : ',dftest[0])
    print('2. P-Value : ',dftest[1])
    print('3. Num of Lags : ',dftest[2])
    print('4. Num of observation used for ADF regression and critical value calculation : ',dftest[3])
    print('5. Critical value')
    for key, val in dftest[4].items():
        print('\t',key,':',val)

ad_test(df['Qty'])


# Uji hipotesis apakah data tersebut stationer atau tidak
# - H0 : data tidak stationer
# - H1 : data statisoner
#     
# Jika
# - p-value > 0.05 maka gagal tolak H0
# - p-value <= 0.05 maka tolak H0
# 
# Nilai p-value = 0, maka kesimpulannya adalah data tersebut stationer

# **AUTOCORRELATION FUNCTION**

# In[10]:


fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,4))

ax1.plot(df_reg['Qty'])
ax1.set_title('Original')
plot_acf(df_reg['Qty'], ax=ax2)


# In[11]:


diff = df_reg['Qty'].diff().dropna()

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,4))

ax1.plot(diff)
ax1.set_title('Diff 1')
plot_acf(diff, ax=ax2)


# In[12]:


diff = df_reg['Qty'].diff().diff().dropna()

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,4))

ax1.plot(diff)
ax1.set_title('Diff 2')
plot_acf(diff, ax=ax2)


# **Nilai D**

# In[13]:


ndiffs(df_reg['Qty'], test='adf')


# Niali D merupakan nilai yang menunjukkan data tersebut stationer atau tidak, jika 0 artinya data tersebut stationer begitupun sebaliknya 

# **Nilai P**

# In[14]:


diff = df_reg['Qty'].diff().dropna()

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,4))

ax1.plot(diff)
ax1.set_title('Diff 1')
plot_pacf(diff, ax=ax2)


# Untuk mendapatkan nilai P secara manual, kita perlu melakukan cek partial autocorrelation yang dimana jika terdapat nilai yang melebihi batas biru, maka nilai tersebut merupakan nilai kandidat dari nilai P

# **Nilai Q**

# In[15]:


diff = df_reg['Qty'].diff().dropna()

fig, (ax1, ax2) = plt.subplots(1,2,figsize=(16,4))

ax1.plot(diff)
ax1.set_title('Diff 1')
plot_acf(diff, ax=ax2)


# Untuk mendapatkan nilai Q secara manual, kita perlu melakukan cek autocorrelation yang dimana jika terdapat nilai yang melebihi batas biru, maka nilai tersebut merupakan nilai kandidat dari nilai Q

# **MENCARI P,D,Q MENGGUNAKAN AUTO ARIMA**

# In[16]:


stepwise_fit = auto_arima(df_reg['Qty'], trace=True, suppress_warnings=True)
stepwise_fit.summary()


# # Modelling

# **MODELLING MENGGUNAKAN ARIMA**

# In[27]:


# Select ratio
ratio = 0.9
 
total_rows = df_reg.shape[0]
train_size = int(total_rows*ratio)
 
# Split data into test and train
train = df_reg[:train_size]
test = df_reg[train_size:]

y = train['Qty']

model = ARIMA(y, order=(22,2,1))
model = model.fit()

y_pred = model.get_forecast(len(test))

y_pred_df = y_pred.conf_int()
y_pred_df['predictions'] = model.predict(start=y_pred_df.index[0], end=y_pred_df.index[-1])
y_pred_df.index = test.index
y_pred_out = y_pred_df['predictions']
# eval(test['Qty'], y_pred_out)
mse = mean_squared_error(test['Qty'], y_pred_out)
print('MSE: ',mse)
rmse = math.sqrt(mse)
print('RMSE: ',rmse)

plt.figure(figsize=(20,5))
plt.plot(train['Qty'])
plt.plot(test['Qty'], color='red')
plt.plot(y_pred_out, color='black', label='ARIMA')
plt.legend()

