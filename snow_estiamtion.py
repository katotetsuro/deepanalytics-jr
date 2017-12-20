
# coding: utf-8

# # todo
# 
# * snowfallデータを追加する
# * n日前までの気象データを追加する
# * 列車の到着時刻 or 走行時間 を追加する（電車によってスピードが違う）
# * 列車のタイプを型番から推測して、追加する（鈍行と特急で違うのでは）
# * xgboostの評価指標にweighted maeを追加する（できないかも？）
# * xgboostのパラメータサーチをhyperoptでがんばる

# In[255]:

import pandas as pd
from glob import glob
import xgboost as xgb
import numpy as np
from matplotlib import pyplot as plt
get_ipython().magic('matplotlib inline')
from datetime import datetime
from pandas.tseries import offsets
import math


# In[3]:

csvs = glob('data/*.csv')


# In[4]:

csvs


# In[34]:

train = pd.read_csv('data/train.csv') 
train['年月日'] = train['年月日'].apply(pd.to_datetime)
test = pd.read_csv('data/test.csv')
test['年月日'] = test['年月日'].apply(pd.to_datetime)


# In[7]:

weather = pd.read_csv('data/weather.csv')
weather['年月日時'] = weather['年月日時'].apply(pd.to_datetime)


# In[9]:

snowfall = pd.read_csv('data/snowfall.csv')
snowfall['年月日時'] = snowfall['年月日時'].apply(pd.to_datetime)


# In[ ]:

snowfall.loc[lambda df: (0 < df['積雪深(軌道)']) & (df['積雪深(軌道)'] < 100), :]


# In[156]:

weather['day'] = weather['年月日時'].apply(lambda x: x.dayofyear)


# In[193]:

s = weather.groupby(['day', '地点'])['気温(℃)'].mean()


# In[ ]:

weather.shift()


# In[213]:

plt.plot(s[:, '富山'], 'b-', alpha=0.5)
plt.plot(s[:, '金沢'], 'r-', alpha=0.5)


# In[212]:

train[ train['停車駅名'] == '金沢']


# In[155]:

d.dayofyear


# In[132]:

weather.keys()


# In[106]:

for k,v in per_train.groups.items():
    plt.plot(train.loc[v.values]['台車部分'].values, c=np.random.rand(3,))


# In[117]:

weather['年月日時']


# In[409]:

#　とりあえずすごいシンプルに、 { (金沢, 富山) x (1日の平均気温, 1日の合計降水量, 1日の合計降雪量, 1日の合計積雪量, 日照時間合計) } x (おととい、昨日、今日)
# くらいのデータでxgboostしてみよかな
train_df = train[train['停車駅名'] == '富山'][['年月日', '列車番号', '台車部分']]


# In[415]:

month = lambda x: x.month
train_df = train_df[ (1<=train_df['年月日'].apply(month)) & (train_df['年月日'].apply(month) <= 3) ]


# In[416]:

len(train_df)


# In[417]:

# train_dfに足してくぞー
diagram = pd.read_csv('data/diagram.csv')
depature_times = pd.to_datetime(diagram[train_df['列車番号']].loc[0])
train_df['出発時刻'] =  depature_times.values
train_df['出発時刻_in_day'] = train_df['出発時刻'].apply(lambda x: x.hour + x.minute/60)
train_df = train_df.rename(columns={'年月日':'day'})


# In[418]:

len(train_df)


# In[419]:

weather['day'] = weather['年月日時'].apply(lambda x: pd.to_datetime(x.strftime('%Y/%m/%d')))


# In[420]:

temperature = weather[['気温(℃)','積雪(cm)', '地点', 'day']].groupby(['day', '地点']).mean().reset_index()
temperature = temperature[ temperature['地点'] != '糸魚川' ]
temperature = temperature.pivot_table(values=['気温(℃)', '積雪(cm)'], index=['day'], columns=['地点'])


# In[421]:

len(temperature)


# In[422]:

features = weather[['降水量(mm)', '降雪(cm)', '日照時間(時間)', '地点', 'day']].groupby(['day', '地点']).sum().reset_index()
features = features[ features['地点'] != '糸魚川' ]
features = features.pivot_table(values=['降水量(mm)', '降雪(cm)'], index=['day'], columns=['地点'])


# In[423]:

len(features)


# In[424]:

train_df = pd.merge(train_df, features.reset_index(), on=['day'])
train_df = pd.merge(train_df, temperature.reset_index(), on=['day'])


# In[425]:

len(train_df)


# In[467]:

all_keys = train_df.keys()
train_x_keys = all_keys.drop(['台車部分', 'day', '列車番号', '出発時刻'])
train_x = train_df[train_x_keys]
train_y = train_df['台車部分']


# In[468]:

dm = xgb.DMatrix(train_x, label=train_y)


# In[478]:

cv = xgb.cv({'eval_metric': 'mae'}, dm, num_boost_round=50)


# In[481]:

cv.sort_values(by='test-mae-mean')


# In[484]:

bst = xgb.train({'eval_metric': 'mae'}, dm, num_boost_round=50)


# In[488]:

len(test)


# In[489]:

test.keys()


# In[495]:

test_df = test
depature_times = pd.to_datetime(diagram[test_df['列車番号']].loc[0])
test_df['出発時刻'] =  depature_times.values
test_df['出発時刻_in_day'] = test_df['出発時刻'].apply(lambda x: x.hour + x.minute/60)
test_df = test_df.rename(columns={'年月日':'day'})
test_df = pd.merge(test_df, features.reset_index(), on=['day'])
test_df = pd.merge(test_df, temperature.reset_index(), on=['day'])


# In[496]:

test_x = test_df[train_x_keys]


# In[498]:

xgb.DMatrix(test_x)


# In[501]:

predict = bst.predict(xgb.DMatrix(test_x))


# In[503]:

plt.plot(predict)


# In[514]:

predict[predict<=0.000007] = 0


# In[517]:

predict_df = pd.DataFrame(data=predict)


# In[519]:

predict_df.to_csv('prediction.csv', header=False)


# In[520]:

train.describe()


# In[521]:

predict_df.describe()


# In[499]:

train_x.keys()


# In[433]:

ari = train_df[ train_df['台車部分'] > 0 ]
nashi = train_df[ train_df['台車部分'] == 0 ]


# In[436]:

ari.describe()


# In[437]:

nashi.describe()

