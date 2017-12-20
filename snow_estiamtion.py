
# coding: utf-8

# # todo
# 
# * snowfallデータを追加する
# * n日前までの気象データを追加する
# * 列車の到着時刻 or 走行時間 を追加する（電車によってスピードが違う）
# * 列車のタイプを型番から推測して、追加する（鈍行と特急で違うのでは）
# * xgboostの評価指標にweighted maeを追加する（できないかも？）
# * xgboostのパラメータサーチをhyperoptでがんばる

# In[36]:

import pandas as pd
from glob import glob
import xgboost as xgb
import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
from datetime import datetime
from pandas.tseries import offsets
import math
from os.path import join


# In[2]:

csvs = glob('data/*.csv')


# In[3]:

csvs


# In[4]:

train = pd.read_csv('data/train.csv') 
train['年月日'] = train['年月日'].apply(pd.to_datetime)
test = pd.read_csv('data/test.csv')
test['年月日'] = test['年月日'].apply(pd.to_datetime)


# In[5]:

weather = pd.read_csv('data/weather.csv')
weather['年月日時'] = weather['年月日時'].apply(pd.to_datetime)


# In[6]:

snowfall = pd.read_csv('data/snowfall.csv')
snowfall['年月日時'] = snowfall['年月日時'].apply(pd.to_datetime)


# In[7]:

#zweather['day'] = weather['年月日時'].apply(lambda x: x.dayofyear)


# In[8]:

s = weather.groupby(['day', '地点'])['気温(℃)'].mean()


# In[9]:

plt.plot(s[:, '富山'], 'b-', alpha=0.5)
plt.plot(s[:, '金沢'], 'r-', alpha=0.5)


# In[12]:

#　とりあえずすごいシンプルに、 { (金沢, 富山) x (1日の平均気温, 1日の合計降水量, 1日の合計降雪量, 1日の合計積雪量, 日照時間合計) } x (おととい、昨日、今日)
# くらいのデータでxgboostしてみよかな
train_df = train[train['停車駅名'] == '富山'][['年月日', '列車番号', '台車部分']]


# In[13]:

month = lambda x: x.month
train_df = train_df[ (1<=train_df['年月日'].apply(month)) & (train_df['年月日'].apply(month) <= 3) ]


# In[14]:

len(train_df)


# In[15]:

# train_dfに足してくぞー
diagram = pd.read_csv('data/diagram.csv')
depature_times = pd.to_datetime(diagram[train_df['列車番号']].loc[0])
train_df['出発時刻'] =  depature_times.values
train_df['出発時刻_in_day'] = train_df['出発時刻'].apply(lambda x: x.hour + x.minute/60)
train_df = train_df.rename(columns={'年月日':'day'})


# In[16]:

len(train_df)


# In[17]:

weather['day'] = weather['年月日時'].apply(lambda x: pd.to_datetime(x.strftime('%Y/%m/%d')))


# In[18]:

temperature = weather[['気温(℃)','積雪(cm)', '地点', 'day']].groupby(['day', '地点']).mean().reset_index()
temperature = temperature[ temperature['地点'] != '糸魚川' ]
temperature = temperature.pivot_table(values=['気温(℃)', '積雪(cm)'], index=['day'], columns=['地点'])


# In[19]:

len(temperature)


# In[20]:

features = weather[['降水量(mm)', '降雪(cm)', '日照時間(時間)', '地点', 'day']].groupby(['day', '地点']).sum().reset_index()
features = features[ features['地点'] != '糸魚川' ]
features = features.pivot_table(values=['降水量(mm)', '降雪(cm)'], index=['day'], columns=['地点'])


# In[21]:

len(features)


# In[22]:

train_df = pd.merge(train_df, features.reset_index(), on=['day'])
train_df = pd.merge(train_df, temperature.reset_index(), on=['day'])


# In[23]:

len(train_df)


# In[24]:

all_keys = train_df.keys()
train_x_keys = all_keys.drop(['台車部分', 'day', '列車番号', '出発時刻'])
train_x = train_df[train_x_keys]
train_y = train_df['台車部分']


# In[25]:

dm = xgb.DMatrix(train_x, label=train_y)


# In[26]:

cv = xgb.cv({'eval_metric': 'mae'}, dm, num_boost_round=50)


# In[27]:

cv.sort_values(by='test-mae-mean')


# In[28]:

bst = xgb.train({'eval_metric': 'mae'}, dm, num_boost_round=50)


# In[29]:

len(test)


# In[30]:

test.keys()


# In[31]:

test_df = test
depature_times = pd.to_datetime(diagram[test_df['列車番号']].loc[0])
test_df['出発時刻'] =  depature_times.values
test_df['出発時刻_in_day'] = test_df['出発時刻'].apply(lambda x: x.hour + x.minute/60)
test_df = test_df.rename(columns={'年月日':'day'})
test_df = pd.merge(test_df, features.reset_index(), on=['day'])
test_df = pd.merge(test_df, temperature.reset_index(), on=['day'])


# In[32]:

test_x = test_df[train_x_keys]


# In[34]:

predict = bst.predict(xgb.DMatrix(test_x))


# In[35]:

plt.plot(predict)


# In[37]:

predict[predict<=0.000007] = 0


# In[38]:

predict_df = pd.DataFrame(data=predict)


# In[39]:

predict_df.to_csv(join('result', 'prediction.csv'), header=False)


# In[ ]:



