import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymysql
from sqlalchemy import create_engine
import time
from sklearn.preprocessing import StandardScaler, normalize, RobustScaler, MinMaxScaler, FunctionTransformer
from scipy.stats import boxcox
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import scipy.cluster.hierarchy as hc
from sklearn.cluster import AgglomerativeClustering
from haversine import haversine, Unit

start_time = time.time()
con = pymysql.connect(host='localhost', user='jiwoo', password='1234', db='TESTDB', charset='utf8', autocommit=True, cursorclass=pymysql.cursors.DictCursor)
cur = con.cursor()
sql = "SELECT pk, 시도, official_price, 경도, 위도, closest_sub, closest_high, school_1km, area, floor, actual FROM off_act_price WHERE 시도='충청북도'"
cur.execute(sql)
rows = cur.fetchall()
con.close()

gangwon = pd.DataFrame(rows)
end_time = time.time()
taketime = str(end_time - start_time)
print("서울특별시 import 실행 소요시간 (단위 초) : " + taketime)

real_trainInfo_pk = gangwon.dropna(subset='actual')
pk_train = real_trainInfo_pk['pk']
real_trainInfo = real_trainInfo_pk.drop(['pk', '시도'], axis=1)
needPredict_pk = gangwon[gangwon['actual'].isnull()==True]
pk_predict = needPredict_pk['pk']
needPredict = needPredict_pk.drop(['pk', '시도'], axis=1)


X = real_trainInfo.drop('actual', axis=1)
realX = needPredict.drop('actual', axis=1)
y = real_trainInfo['actual']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=1)
train_data = x_train.join(y_train)

train_data['floor'] = train_data['floor']+2
needPredict['floor'] = needPredict['floor']+2
#train_data['1km내sub수+'] = train_data['1km내sub수']+2
train_data['school_1km'] = train_data['school_1km']+2
needPredict['school_1km'] = needPredict['school_1km']+2

from scipy.stats import boxcox
#train_data['area'] = train_data['area'].apply(lambda x: boxcox(x)[0])
train_data[['school_1km', 'area']] = np.sqrt(train_data[['school_1km', 'area']])
train_data[['actual','floor', 'official_price', 'closest_sub', 'closest_high']] = np.log(train_data[['actual','floor', 'official_price', 'closest_sub', 'closest_high']])
#needPredict['area'] = needPredict['area'].apply(lambda x: boxcox(x)[0])
needPredict[['school_1km', 'area']] = np.sqrt(needPredict[['school_1km', 'area']])
needPredict[['floor', 'official_price', 'closest_sub', 'closest_high']] = np.log(needPredict[['floor', 'official_price', 'closest_sub', 'closest_high']])

x_train, y_train = train_data.drop(['actual'], axis=1), train_data['actual']

valid_data = x_valid.join(y_valid)

valid_data['floor'] = valid_data['floor']+2
needPredict['floor'] = needPredict['floor']+2
#valid_data['1km내sub수+'] = valid_data['1km내sub수']+2
valid_data['school_1km'] = valid_data['school_1km']+2
needPredict['school_1km'] = needPredict['school_1km']+2

#valid_data['area'] = valid_data['area'].apply(lambda x: boxcox(x)[0])
valid_data[['school_1km', 'area']] = np.sqrt(valid_data[['school_1km', 'area']])
valid_data[['actual','floor', 'official_price', 'closest_sub', 'closest_high']] = np.log(valid_data[['actual','floor', 'official_price', 'closest_sub', 'closest_high']])
#needPredict['area'] = needPredict['area'].apply(lambda x: boxcox(x)[0])
needPredict[['school_1km', 'area']] = np.sqrt(needPredict[['school_1km', 'area']])
needPredict[['floor', 'official_price', 'closest_sub', 'closest_high']] = np.log(needPredict[['floor', 'official_price', 'closest_sub', 'closest_high']])

x_valid, y_valid = valid_data.drop(['actual'], axis=1), valid_data['actual']

test_data = X_test.join(y_test)

test_data['floor'] = test_data['floor']+2
needPredict['floor'] = needPredict['floor']+2
#valid_data['1km내sub수+'] = valid_data['1km내sub수']+2
test_data['school_1km'] = test_data['school_1km']+2
needPredict['school_1km'] = needPredict['school_1km']+2

#valid_data['area'] = valid_data['area'].apply(lambda x: boxcox(x)[0])
test_data[['school_1km', 'area']] = np.sqrt(test_data[['school_1km', 'area']])
test_data[['actual','floor', 'official_price', 'closest_sub', 'closest_high']] = np.log(test_data[['actual','floor', 'official_price', 'closest_sub', 'closest_high']])
#needPredict['area'] = needPredict['area'].apply(lambda x: boxcox(x)[0])
needPredict[['school_1km', 'area']] = np.sqrt(needPredict[['school_1km', 'area']])
needPredict[['floor', 'official_price', 'closest_sub', 'closest_high']] = np.log(needPredict[['floor', 'official_price', 'closest_sub', 'closest_high']])

X_test, y_test = test_data.drop(['actual'], axis=1), test_data['actual']

scaler = MinMaxScaler()

scaler.fit(x_train)
x_train_s = scaler.transform(x_train)
x_valid_s = scaler.transform(x_valid)
X_test_s = scaler.transform(X_test)

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

tree = DecisionTreeRegressor()
tree.fit(x_train_s, y_train)
print('====== 의사결정나무 ======')
treeScore = tree.score(X_test_s, y_test)
print('tree.score = ', end=' ')
print(treeScore)

y_pred_tr = tree.predict(x_train_s)

tree_R2 = r2_score(y_train, y_pred_tr)
tree_mae = mean_absolute_error(y_train, y_pred_tr)
print('Train Set R2 score : ', end=' ')
print(tree_R2)
print('Train Set MAE : ', end=' ')
print(tree_mae)

y_pred_tr = tree.predict(x_valid_s)

tree_R2 = r2_score(y_valid, y_pred_tr)
tree_mae = mean_absolute_error(y_valid, y_pred_tr)
print('Valid Set R2 score : ', end=' ')
print(tree_R2)
print('Valid Set MAE : ', end=' ')
print(tree_mae)

def display_scores(model, scores):
    print('<<', model, '모델 평가 결과 >>')
    print('평균 RMSE: ', scores.mean())
    print('표준편차: ', scores.std())

from sklearn.model_selection import cross_val_score
tree_scores = cross_val_score(tree, x_valid_s, y_valid, scoring='neg_mean_squared_error', cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)
display_scores('의사결정나무', tree_rmse_scores)

final_pred_tree = tree.predict(X_test_s)

from sklearn.metrics import mean_squared_error
final_mse_tree = mean_squared_error(y_test, final_pred_tree)
final_rmse_tree = np.sqrt(final_mse_tree)
final_r2_tree = r2_score(y_test, final_pred_tree)

print('Test Set RMSE: ', final_rmse_tree)
print('Test Set R2: ', final_r2_tree)

needPredict_s = scaler.transform(realX)
finalPred = tree.predict(needPredict_s)

# 예측값
pred_tr = pd.DataFrame(final_pred_tree, columns=['예측가격'])
pred_tr['predicted_price'] = np.exp(pred_tr['예측가격'])
actual_tr = pd.DataFrame(y_test)
actual_tr['actual_price'] = np.exp(actual_tr['actual'])
actual_tr.reset_index(inplace=True, drop=True)
table_tr = pd.concat([pred_tr, actual_tr], axis=1)
final_tr = table_tr[['predicted_price', 'actual_price']]

plt.rc('font', family='NanumGothic')
final_tr.iloc[1700:1800, :].plot(figsize=(30,7))
plt.show()

#pred_tr['predicted_price'] = np.exp(pred_tr['예측가격'])
#table_tr = pd.concat([needPredict, pred_tr], axis=1)
#final_table = pd.concat([pk_predict, table_tr], axis=1)


pd.set_option('display.max_columns', None)
print(final_tr)

#subtract = gangwon.shape[0] - gangwon['actual'].isnull().sum()
#print('<<< '+gangwon['시도'].unique()+' >>>')
#print('총 행수')
#print(gangwon.shape[0])
#print('')
#print('유효 실거래가 수')
#print(subtract)

#print(gangwon.isnull().sum())

#engine = create_engine("mysql+pymysql://jiwoo:1234@localhost:3306/TESTDB?charset=utf8mb4")
#conn = engine.connect()
#table_name = 'off_act_price'
#start_time = time.time()
#total.to_sql(name=table_name, con=engine, if_exists='append', index=False)
#end_time = time.time()

#taketime = str(end_time - start_time)
#print("강원도 스크립트 실행 소요시간 (단위 초) : " + taketime)
