
# coding: utf-8

# # todo
# 
# * snowfallデータを追加する
# * 列車のタイプを型番から推測して、追加する（鈍行と特急で違うのでは）
# * xgboostの評価指標にweighted maeを追加する（できないかも？）
# * xgboostのパラメータサーチをhyperoptでがんばる
# 
# # done
# * n日前までの気象データを追加する
# * 列車の到着時刻 or 走行時間 を追加する（電車によってスピードが違う）

# In[9]:

import pandas as pd
from glob import glob
import xgboost as xgb
import numpy as np
from matplotlib_fname import pyplot as plt
get_ipython().magic('matplotlib inline')
from datetime import datetime
from pandas.tseries import offsets
import math
from os.path import join
from hyperopt import fmin, tpe, hp, STATUS_OK, Trials


# In[10]:

csvs = glob('data/*.csv')


# In[11]:

csvs


# In[92]:

train = pd.read_csv('data/train.csv') 
train['年月日'] = train['年月日'].apply(pd.to_datetime)
test = pd.read_csv('data/test.csv')
test['年月日'] = test['年月日'].apply(pd.to_datetime)


# In[13]:

weather = pd.read_csv('data/weather.csv')
weather['年月日時'] = weather['年月日時'].apply(pd.to_datetime)
weather['day'] = weather['年月日時'].apply(lambda x: x.dayofyear)


# In[14]:

#　とりあえずすごいシンプルに、 { (金沢, 富山) x (1日の平均気温, 1日の合計降水量, 1日の合計降雪量, 1日の合計積雪量, 日照時間合計) } x (おととい、昨日、今日)
# くらいのデータでxgboostしてみよかな
train_df = train[train['停車駅名'] == '富山'][['年月日', '列車番号', '台車部分']]


# In[15]:

# 1年分のデータがあるけど、夏のデータとかあっても意味ないと予想して、test期間と同じ頃のデータだけを使うことにする
month = lambda x: x.month
train_df = train_df[ (1<=train_df['年月日'].apply(month)) & (train_df['年月日'].apply(month) <= 3) ]


# In[16]:

# todo 地点でわける
# train_dfの1行を受け取って、各行に対して[offset_day, offset_day+delta_day]の区間での集計（気温、降雪etc...)を取る
from pandas.tseries.offsets import *

def summarize_weather(row, offset=pd.tseries.offsets.Day(1), delta=pd.tseries.offsets.Day(1)):
    d = row['出発時刻']
    s = d - offset
    e = s + delta
    sample_range_df = weather[ (s<=weather['年月日時']) & (weather['年月日時']<e) ]
#    return pd.concat([sample_range_df[['気温(℃)', '積雪(cm)']].mean(), sample_range_df[['降雪(cm)', '降水量(mm)']].sum()])
    # 2地点毎に集計をわける
    station = ['富山', '金沢']
    series = []
    for s in station:
        df = sample_range_df[ sample_range_df['地点'] == s ]
        # 集計
        df = pd.concat([df[['気温(℃)', '積雪(cm)']].mean(), df[['降雪(cm)', '降水量(mm)']].sum()])
        # 名前に地点,offset_day, delta_dayでprefixをつけます
        prefix = '({}, {})_{}_'.format(offset.freqstr, delta.freqstr, s)
        new_names = {c: prefix + c for c in df.index}
        df = df.rename(index=new_names)
        series.append(df)
    return pd.concat(series)


# In[17]:

def summarize_snowfall(row, offset=pd.tseries.offsets.Day(1), delta=pd.tseries.offsets.Day(1)):
    d = row['出発時刻']
    s = d - offset
    e = s + delta
    sample_range_df = snowfall[ (s<=snowfall['年月日時']) & (snowfall['年月日時']<e) ]
    # 地点毎に集計をわける
    locations = ['東金沢', '津幡', '小矢部', '高岡赤祖父', '富山野々上']
    series = []
    for s in locations:
        df = sample_range_df[ sample_range_df['地域名'] == s ]
        # 集計
        df = df[['積雪深(軌道)', '積雪深(側溝)', '積雪深(軌道A)', '積雪深(軌道B)']].mean()
        # 名前に地点,offset_day, delta_dayでprefixをつけます
        prefix = '(降雪データ_{}, {})_{}_'.format(offset.freqstr, delta.freqstr, s)
        new_names = {c: prefix + c for c in df.index}
        df = df.rename(index=new_names)
        series.append(df)
    return pd.concat(series)

def clip_values(row):
    columns = ['積雪深(軌道)', '積雪深(側溝)', '積雪深(軌道A)', '積雪深(軌道B)']
    for c in columns:
        row[c] = max(min(row[c], 100), 0)
    return row

snowfall = pd.read_csv('data/snowfall.csv')
snowfall['年月日時'] = snowfall['年月日時'].apply(pd.to_datetime)
snowfall['day'] = snowfall['年月日時'].apply(lambda x: pd.to_datetime(x.strftime('%Y-%m-%d')))
#snowfall = snowfall[ (train['年月日'].min() <= snowfall['年月日時']) & (snowfall['年月日時'] <= train['年月日'].max() ) ]
snowfall = snowfall.apply(clip_values, axis=1)


# In[18]:

def calc_features(dataset):
    # 出発時刻を24時間表記の連続量で表す(小数点込み)
    diagram = pd.read_csv('data/diagram.csv')
    depature_times = pd.to_datetime(diagram[dataset['列車番号']].loc[0])
    dataset['出発時刻'] =  depature_times.values
    dataset['出発時刻_in_day'] = dataset['出発時刻'].apply(lambda x: x.hour + x.minute/60)
    dataset = dataset.rename(columns={'年月日':'day'})
    
    # 富山に着くまでに何時間走ったか
    def elapsed(row):
        x = pd.to_datetime(diagram[row['列車番号']].loc[2]) - pd.to_datetime(diagram[row['列車番号']].loc[0])
        return x.seconds/60/60
    dataset['走行時間'] = dataset.apply(elapsed, axis=1)
    
    # 出発時刻の年月日がおかしいのをなおす 出発時刻のdiagram.csvには年月日が入ってないので、
    # pd.to_datetimeで変換すると年月日が実行した日になっちゃう
    # それをtrain_dfのdayから補間したいんだけど、いいやり方がわからんから文字列型のtimestampを作ってしまうことにした
    for i, row in dataset.iterrows():
        str_date = row['day'].strftime('%Y-%m-%d') + row['出発時刻'].strftime(' %H:%M:%S')
        dataset.loc[i, '出発時刻'] = pd.to_datetime(str_date)
        
    # 列車番号によってなんかタイプ分けがされてるんだよなー.これが効果あるかわからんけども
    # train_typesは こんなかんじでトレーニングデータから取得
    # train_types = list(set(map(lambda x: x[0], train_df.groupby('列車番号').groups.keys())))
    train_types = ['5', '8', '3', '9']
    dataset['train_type'] = dataset.apply(lambda row: train_types.index(row['列車番号'][0]), axis=1)
    
    # 金沢駅を出るときに除雪済みの車両にフラグを立てる
    nosnow_trains = pd.read_csv('data/kanazawa_nosnow.csv').values.reshape(-1)
    dataset['金沢出発時除雪済み'] = dataset.apply(lambda x: 1 if x['列車番号'] in nosnow_trains else 0, axis=1)
    
    # time_slicesで指定した形で、天候情報を整形する
    # 出発時刻のN時間前から、　M時間の間の集計 という形
    time_slices = [
        (Hour(6), Hour(6)),
        (Hour(12), Hour(6)),
        (Hour(18), Hour(6)),
        (Hour(24), Hour(6))
    ]
    tmp = [dataset.apply(lambda x: summarize_weather(x, offset=t[0], delta=t[1]), axis=1)  for t in time_slices]
    dataset = pd.concat( [dataset, pd.concat(tmp, axis=1)], axis=1)
    
    # JR独自の降雪データを使う
    tmp = dataset.apply(summarize_snowfall, axis=1)
    dataset = pd.concat([dataset, tmp], axis=1)
    
    return dataset


# In[19]:

train_df = calc_features(train_df)


# In[23]:

train_df.columns


# In[20]:

train_df


# In[12]:

# train_dfに足してくぞー
diagram = pd.read_csv('data/diagram.csv')
depature_times = pd.to_datetime(diagram[train_df['列車番号']].loc[0])
train_df['出発時刻'] =  depature_times.values
train_df['出発時刻_in_day'] = train_df['出発時刻'].apply(lambda x: x.hour + x.minute/60)
train_df = train_df.rename(columns={'年月日':'day'})


# In[13]:

# 何秒走ったか
def elapsed(row):
    x = pd.to_datetime(diagram[row['列車番号']].loc[2]) - pd.to_datetime(diagram[row['列車番号']].loc[0])
    return x.seconds/60/60
#train_df['走行時間'] = (pd.to_datetime(diagram[train_df['列車番号']].loc[2]) - pd.to_datetime(diagram[train_df['列車番号']].loc[0])).apply(lambda x: x.seconds/60/60).values
train_df['走行時間'] = train_df.apply(elapsed, axis=1)


# In[14]:

# 出発時刻の年月日がおかしいのをなおす 出発時刻のdiagram.csvには年月日が入ってないので、
# pd.to_datetimeで変換すると年月日が実行した日になっちゃう
# それをtrain_dfのdayから補間したいんだけど、いいやり方がわからんから文字列型のtimestampを作ってしまうことにした
for i, row in train_df.iterrows():
    str_date = row['day'].strftime('%Y-%m-%d') + row['出発時刻'].strftime(' %H:%M:%S')
    train_df.loc[i, '出発時刻'] = pd.to_datetime(str_date)


# In[15]:

weather['day'] = weather['年月日時'].apply(lambda x: pd.to_datetime(x.strftime('%Y/%m/%d')))


# In[ ]:

temperature = weather[['気温(℃)','積雪(cm)', '地点', 'day']].groupby(['day', '地点']).mean().reset_index()
temperature = temperature[ temperature['地点'] != '糸魚川' ]
temperature = temperature.pivot_table(values=['気温(℃)', '積雪(cm)'], index=['day'], columns=['地点'])


# In[ ]:

len(temperature)


# In[ ]:

features = weather[['降水量(mm)', '降雪(cm)', '日照時間(時間)', '地点', 'day']].groupby(['day', '地点']).sum().reset_index()
features = features[ features['地点'] != '糸魚川' ]
features = features.pivot_table(values=['降水量(mm)', '降雪(cm)'], index=['day'], columns=['地点'])


# In[ ]:

len(features)


# In[ ]:

train_df = pd.merge(train_df, features.reset_index(), on=['day'])
train_df = pd.merge(train_df, temperature.reset_index(), on=['day'])


# In[ ]:

len(train_df)


# In[ ]:

delta_day = 1
offset_day = 1
for d in train_df['出発時刻']:
    s = d - pd.tseries.offsets.Day(offset_day)
    e = s + pd.tseries.offsets.Day(delta_day)
    sample_range_df = weather[ (s<=weather['年月日時']) & (weather['年月日時']<e) ]
    


# In[17]:

#出発時刻のちょっとした違いで24時間以内の気温とかが変化するんだぞということを確かめる
# 6時間違うと結構かわってくる
d = train_df['出発時刻'][0]
print(d)
s = d - pd.tseries.offsets.Day(1)
e = s + pd.tseries.offsets.Day(1)
sample_range_df = weather[ (s<=weather['年月日時']) & (weather['年月日時']<e) ]
sample_range_df[['気温(℃)', '積雪(cm)']].mean(), sample_range_df[['降雪(cm)', '降水量(mm)']].sum()


# In[18]:

d = train_df['出発時刻'][1]
print(d)
s = d - pd.tseries.offsets.Day(1)
e = s + pd.tseries.offsets.Day(1)
sample_range_df = weather[ (s<=weather['年月日時']) & (weather['年月日時']<e) ]
sample_range_df[['気温(℃)', '積雪(cm)']].mean(), sample_range_df[['降雪(cm)', '降水量(mm)']].sum()


# In[59]:

# 列車番号によってなんかタイプ分けがされてるんだよなー.これが効果あるかわからんけども
train_types = list(set(map(lambda x: x[0], train_df.groupby('列車番号').groups.keys())))
train_df['train_type'] = train_df.apply(lambda row: train_types.index(row['列車番号'][0]), axis=1)


# In[66]:

nosnow_trains = pd.read_csv('data/kanazawa_nosnow.csv').values.reshape(-1)


# In[70]:

# 除雪済みフラグ
train_df['金沢出発時除雪済み'] = train_df.apply(lambda x: 1 if x['列車番号'] in nosnow_trains else 0, axis=1)


# In[21]:

# todo 地点でわける
# train_dfの1行を受け取って、各行に対して[offset_day, offset_day+delta_day]の区間での集計（気温、降雪etc...)を取る
from pandas.tseries.offsets import *

def summarize_weather(row, offset=pd.tseries.offsets.Day(1), delta=pd.tseries.offsets.Day(1)):
    d = row['出発時刻']
    s = d - offset
    e = s + delta
    sample_range_df = weather[ (s<=weather['年月日時']) & (weather['年月日時']<e) ]
#    return pd.concat([sample_range_df[['気温(℃)', '積雪(cm)']].mean(), sample_range_df[['降雪(cm)', '降水量(mm)']].sum()])
    # 2地点毎に集計をわける
    station = ['富山', '金沢']
    series = []
    for s in station:
        df = sample_range_df[ sample_range_df['地点'] == s ]
        # 集計
        df = pd.concat([df[['気温(℃)', '積雪(cm)']].mean(), df[['降雪(cm)', '降水量(mm)']].sum()])
        # 名前に地点,offset_day, delta_dayでprefixをつけます
        prefix = '({}, {})_{}_'.format(offset.freqstr, delta.freqstr, s)
        new_names = {c: prefix + c for c in df.index}
        df = df.rename(index=new_names)
        series.append(df)
    return pd.concat(series)


# In[ ]:

# d = train_df.loc[0]['出発時刻']
# s = d - pd.tseries.offsets.Day(1)
# e = s + pd.tseries.offsets.Day(1)
# sample_range_df = weather[ (s<=weather['年月日時']) & (weather['年月日時']<e) ]
# station = ['富山', '金沢']

# series = []
# for s in station:
#     df = sample_range_df[ sample_range_df['地点'] == s ]
#     # 集計
#     df = pd.concat([df[['気温(℃)', '積雪(cm)']].mean(), df[['降雪(cm)', '降水量(mm)']].sum()])
#     #series.append(df[['気温(℃)', '積雪(cm)']].mean())
#     #series.append(df[['降雪(cm)', '降水量(mm)']].sum())
#     # 名前に地点をprefixでつけます
#     new_names = {c: s + '_' + c for c in df.index}
#     df = df.rename(index=new_names)
#     series.append(df)
# print(pd.concat(series))


# In[22]:

d = train_df.loc[0]['出発時刻']
s = d - pd.tseries.offsets.Day(1)
print(s, s + Hour(5))


# In[23]:

tmp = train_df.apply(summarize_weather, axis=1)
tmp2 = train_df.apply(lambda x: summarize_weather(x, offset=Day(2), delta=Hour(6)), axis=1)


# In[ ]:

from pandas.tseries.offsets import *
train_df.apply(lambda x: summarize_weather(x, offset=Day(2), delta=Hour(6)), axis=1).loc[0]


# In[ ]:

train_df.apply(lambda x: summarize_weather(x, Day(2)+Hour(6), delta=Hour(6)), axis=1).loc[0]


# In[161]:

time_slices = [
    (Hour(6), Hour(6)),
    (Hour(12), Hour(6)),
    (Hour(18), Hour(6)),
    (Hour(24), Hour(6))
]
tmp = [train_df.apply(lambda x: summarize_weather(x, offset=t[0], delta=t[1]), axis=1)  for t in time_slices]
train_df_2 = pd.concat( [train_df, pd.concat(tmp, axis=1)], axis=1)


# In[162]:

def clip_values(row):
    columns = ['積雪深(軌道)', '積雪深(側溝)', '積雪深(軌道A)', '積雪深(軌道B)']
    for c in columns:
        row[c] = max(min(row[c], 100), 0)
    return row

snowfall = snowfall.apply(clip_values, axis=1)

def summarize_snowfall(row, offset=pd.tseries.offsets.Day(1), delta=pd.tseries.offsets.Day(1)):
    d = row['出発時刻']
    s = d - offset
    e = s + delta
    sample_range_df = snowfall[ (s<=snowfall['年月日時']) & (snowfall['年月日時']<e) ]
    # 地点毎に集計をわける
    locations = ['東金沢', '津幡', '小矢部', '高岡赤祖父', '富山野々上']
    series = []
    for s in locations:
        df = sample_range_df[ sample_range_df['地域名'] == s ]
        # 集計
        df = df[['積雪深(軌道)', '積雪深(側溝)', '積雪深(軌道A)', '積雪深(軌道B)']].mean()
        # 名前に地点,offset_day, delta_dayでprefixをつけます
        prefix = '(降雪データ_{}, {})_{}_'.format(offset.freqstr, delta.freqstr, s)
        new_names = {c: prefix + c for c in df.index}
        df = df.rename(index=new_names)
        series.append(df)
    return pd.concat(series)

tmp = train_df.apply(summarize_snowfall, axis=1)
train_df_2 = pd.concat([train_df_2, tmp], axis=1)


# In[ ]:

tmp.loc[0], tmp2.loc[0]


# In[164]:

train_df_2.keys()


# In[25]:

all_keys = train_df.keys()
train_x_keys = all_keys.drop(['台車部分', 'day', '列車番号', '出発時刻'])
knzw = list(filter(lambda x: '金沢' in x, all_keys))
#train_x_keys = train_x_keys.drop(knzw)

train_x = train_df[train_x_keys]
train_y = train_df['台車部分']


# In[26]:

dm = xgb.DMatrix(train_x, label=train_y)


# In[125]:

best_parameters = {'eval_metric':'mae', 'colsample_bytree': 0.65, 'eta': 0.30000000000000004, 'gamma': 0.0, 'max_depth': 3, 'min_child_weight': 1.0, 'subsample': 1.0}
cv = xgb.cv(best_parameters, dm, num_boost_round=42)


# In[126]:

cv.sort_values(by='test-mae-mean')[:5]


# In[127]:

bst = xgb.train(best_parameters, dm, num_boost_round=42)


# In[128]:

xgb.plot_importance(bst, max_num_features=20)


# In[93]:

test.keys()


# In[94]:

test_df = calc_features(test)


# In[96]:

test_x = test_df[train_x_keys]


# In[104]:

predict = bst.predict(xgb.DMatrix(test_x))


# In[105]:

predict


# In[109]:

plt.plot(predict)


# In[69]:

plt.plot(predict)


# In[108]:

predict[predict<=9.55462456e-05] = 0


# In[110]:

predict_df = pd.DataFrame(data=predict)


# In[111]:

predict_df.to_csv(join('result', 'prediction_clipped.csv'), header=False)


# In[102]:

def clip_values(row):
    columns = ['積雪深(軌道)', '積雪深(側溝)', '積雪深(軌道A)', '積雪深(軌道B)']
    for c in columns:
        row[c] = max(min(row[c], 100), 0)
    return row


# In[111]:

locations_df = locations_df.apply(clip_values, axis=1)


# In[109]:

locations_df['day'] = locations_df['年月日時'].apply(lambda x: pd.to_datetime(x).strftime('%Y/%m/%d'))


# In[115]:

locations_df.set_index(['day', '地域名']).groupby(level=[0,1]).mean()


# In[62]:

# hyperopt挑戦
from sklearn.model_selection import ShuffleSplit
from sklearn import datasets#, cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, log_loss
from sklearn.cross_validation import KFold
from sklearn import preprocessing


# In[115]:

#Split data set to train and test data
X_train, X_test, y_train, y_test = train_test_split(
     train_x.values, train_y.values, test_size=0.5, random_state=25)
np.random.seed(25)


# In[120]:

def score(params):
    print("Training with params : ")
    print(params)
    N_boost_round=[]
    Score=[]
    skf = ShuffleSplit(n_splits=10, random_state=25)
    for train, test in skf.split(X_train):
        X_Train, X_Test, y_Train, y_Test = X_train[train], X_train[test], y_train[train], y_train[test]
        dtrain = xgb.DMatrix(X_Train, label=y_Train)
        dvalid = xgb.DMatrix(X_Test, label=y_Test)
        watchlist = [(dtrain, 'train'),(dvalid, 'eval')]
        model = xgb.train(params, dtrain, num_boost_round=150,evals=watchlist,early_stopping_rounds=10)
        predictions = model.predict(dvalid)
        N = model.best_iteration
        N_boost_round.append(N)
        score = model.best_score
        Score.append(score)
    Average_best_num_boost_round = np.average(N_boost_round)
    Average_best_score = np.average(Score)
    print("\tAverage of best iteration {0}\n".format(Average_best_num_boost_round))
    print("\tScore {0}\n\n".format(Average_best_score))
    return {'loss': Average_best_score, 'status': STATUS_OK}


def optimize(trials):
    space = {
        'eval_metric': 'mae',
        
        #Control complexity of model
        "eta" : hp.quniform("eta", 0.2, 0.6, 0.05),
        "max_depth" : hp.choice("max_depth", range(1, 10)),
        "min_child_weight" : hp.quniform('min_child_weight', 1, 10, 1),
        'gamma' : hp.quniform('gamma', 0, 1, 0.05),
        
        #Improve noise robustness 
        "subsample" : hp.quniform('subsample', 0.5, 1, 0.05),
        "colsample_bytree" : hp.quniform('colsample_bytree', 0.5, 1, 0.05),
        
        'silent' : 1}
    best = fmin(score, space, algo=tpe.suggest, trials=trials, max_evals=250)
    print("best parameters",best)

trials = Trials()
ret = optimize(trials)


# In[83]:

#出力
#best parameters {'colsample_bytree': 0.55, 'eta': 0.30000000000000004, 'gamma': 0.0, 'max_depth': 4, 'min_child_weight': 1.0, 'subsample': 0.8500000000000001}

#Adapt best params
params = {
    'colsample_bytree': 0.55,
    'eta': 0.30000000000000004,
    'gamma': 0.0,
    'max_depth': 4,
    'min_child_weight': 1.0,
    'subsample': 0.8500000000000001
}

score(params)


# In[119]:

best_parameters


# In[ ]:



