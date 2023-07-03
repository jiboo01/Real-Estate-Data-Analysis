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
sql = "SELECT pk, 시도, official_price, 경도, 위도, closest_sub, closest_high, school_1km, area, floor, actual FROM off_act_price WHERE 시도='강원도'"
cur.execute(sql)
rows = cur.fetchall()
con.close()

gangwon = pd.DataFrame(rows)
end_time = time.time()
taketime = str(end_time - start_time)
print("강원도 import 실행 소요시간 (단위 초) : " + taketime)

real_trainInfo = gangwon.dropna(subset='actual')

needPredict = gangwon[gangwon['actual'].isnull()==True]



X = real_trainInfo.drop('actual', axis=1)
y = real_trainInfo['actual']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=1)

train_pk = x_train['pk']
valid_pk = x_valid['pk']
test_pk = X_test['pk']
predict_pk = needPredict['pk']

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

realX = needPredict.drop(['actual', 'pk', '시도'], axis=1)

x_train, y_train = train_data.drop(['actual', 'pk', '시도'], axis=1), train_data['actual']

valid_data = x_valid.join(y_valid)

valid_data['floor'] = valid_data['floor']+2
#valid_data['1km내sub수+'] = valid_data['1km내sub수']+2
valid_data['school_1km'] = valid_data['school_1km']+2

#valid_data['area'] = valid_data['area'].apply(lambda x: boxcox(x)[0])
valid_data[['school_1km', 'area']] = np.sqrt(valid_data[['school_1km', 'area']])
valid_data[['actual','floor', 'official_price', 'closest_sub', 'closest_high']] = np.log(valid_data[['actual','floor', 'official_price', 'closest_sub', 'closest_high']])

x_valid, y_valid = valid_data.drop(['actual', 'pk', '시도'], axis=1), valid_data['actual']

test_data = X_test.join(y_test)

test_data['floor'] = test_data['floor']+2
#valid_data['1km내sub수+'] = valid_data['1km내sub수']+2
test_data['school_1km'] = test_data['school_1km']+2

#valid_data['area'] = valid_data['area'].apply(lambda x: boxcox(x)[0])
test_data[['school_1km', 'area']] = np.sqrt(test_data[['school_1km', 'area']])
test_data[['actual','floor', 'official_price', 'closest_sub', 'closest_high']] = np.log(test_data[['actual','floor', 'official_price', 'closest_sub', 'closest_high']])

y_test_actual_price = test_data['actual']
X_test, y_test = test_data.drop(['actual', 'pk', '시도'], axis=1), test_data['actual']

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
print(' ')
print('====== 의사결정나무 ======')
treeScore = tree.score(X_test_s, y_test)
print('tree.score = ', end=' ')
print(treeScore)

y_pred_train = tree.predict(x_train_s)

tree_R2 = r2_score(y_train, y_pred_train)
tree_mae = mean_absolute_error(y_train, y_pred_train)
print('Train Set R2 score : ', end=' ')
print(tree_R2)
print('Train Set MAE : ', end=' ')
print(tree_mae)

y_pred_valid = tree.predict(x_valid_s)

tree_R2 = r2_score(y_valid, y_pred_valid)
tree_mae = mean_absolute_error(y_valid, y_pred_valid)
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

#needPredict_s = scaler.transform(realX)
#finalPred = tree.predict(needPredict_s)
#needPred_df = pd.DataFrame(finalPred, columns=['예측가격'])
#needPred_df['pk'] = predict_pk.values
#needPred_df['predicted_price'] = np.exp(needPred_df['예측가격'])
#needPred_df.drop('예측가격', axis=1, inplace=True)
#print(' ')
#print('<< null 예측값 >>')
#print(needPred_df)

# 예측값
pred_tr = pd.DataFrame(final_pred_tree, columns=['예측가격'])
pred_tr['pk'] = test_pk.values
pred_tr['predicted_price'] = np.exp(pred_tr['예측가격'])
pred_tr['actual_price_unscaled'] = y_test_actual_price.values
pred_tr['actual_price'] = np.exp(pred_tr['actual_price_unscaled'])
pred_tr = pred_tr.drop('actual_price_unscaled', axis=1)
pred_tr['predicted_price'] = pred_tr['predicted_price'].astype(float)
# 결측치가 있는 것 같지는 않고 nan 이런 식으로 숫자가 아닌 데이터가 있는 듯. 확인해보기 바람
pred_tr.drop('예측가격', axis=1, inplace=True)
#pred_tr['predicted_price'] = pred_tr['predicted_price'].apply(lambda x: '{:.0f}'.format(x))
#jisang['codes'] = jisang['호str'].apply(lambda x: int(x) if x.isnumeric() else 0)

pred_tr['error'] = (pred_tr['actual_price'] - pred_tr['predicted_price']) / pred_tr['actual_price'] * 100
print(' ')
print(pred_tr.describe())

error_values = pred_tr['error']

# Define the custom bin ranges
bin_ranges = [
    (-500, -50), (-50, -40), (-40, -30), (-30, -20), (-20, -10), (-10, 0),
    (0,10), (10,20), (20,30), (30,40), (40,50), (50, 500)
    # Add more custom ranges as needed
]

# Assign bins to data
binned_data = pd.cut(error_values, bins=[range[0] for range in bin_ranges] + [bin_ranges[-1][1]], labels=[i for i in range(len(bin_ranges))])

# Calculate proportions
proportions = binned_data.value_counts(normalize=True).sort_index()

# Plot the histogram
plt.bar([f'{range[0]}% to {range[1]}%' for range in bin_ranges], proportions.values)
plt.xlabel('Error Range')
plt.ylabel('Proportion')
plt.title('Proportion of Errors in Gangwon')
plt.xticks(rotation=45)
plt.grid(True, axis='y', linestyle='--')

# Show the plot
plt.show()

