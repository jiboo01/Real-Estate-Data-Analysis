import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymysql
from sqlalchemy import create_engine
import time
from sklearn.preprocessing import StandardScaler, normalize, RobustScaler, MinMaxScaler, FunctionTransformer
from sklearn.model_selection import StratifiedShuffleSplit

# Measure time taken to import table
start_time = time.time()

# Connect to MySQL database
con = pymysql.connect(host='localhost', user='jiwoo', password='1234', db='TESTDB', charset='utf8', autocommit=True, cursorclass=pymysql.cursors.DictCursor)
cur = con.cursor()

# Data for ML are stored in table 'off_act_price'
# Extracting data by administrative districts; in this case 대전광역시(Daejeon)

# << Columns explained >>

# 시도 : Administrative District in Korea
# official_price : Appraised Price of a certain unit within the apartment(공시지가)
# 경도 : longitude
# 위도 : latitude
# closest_sub : Direct distance to closest subway st. from apartment
# closest_high : Direct distance to closest high school st. from apartment
# school_1km : Number of elementary/middle/high schools within a 1km radius from apartment
# area : Area(pyeong) of a certain unit (pyeong = square meter/3.3)
# floor : Floor that a certain unit is located
# actual : Transaction Price of a certain unit within the apartment(실거래가)

sql = "SELECT pk, 시도, official_price, 경도, 위도, closest_sub, closest_high, school_1km, area, floor, actual FROM off_act_price WHERE 시도='대전광역시'"
cur.execute(sql)
rows = cur.fetchall()
con.close()



# Turn table into a dataframe
daejeon = pd.DataFrame(rows)
end_time = time.time()
taketime = str(end_time - start_time)
print("대전광역시 import time (s) : " + taketime)


# Sort data that have transaction price 
real_trainInfo = daejeon.dropna(subset='actual')


# Sort data that lack of transaction price
needPredict = daejeon[daejeon['actual'].isnull()==True]


# Delete target variable to make train dataset
X = real_trainInfo.drop('actual', axis=1)
y = real_trainInfo['actual']


# Split train/valid/test set to 6:2:2
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
x_train, x_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=1)


# Extract 'pk' column to use as key value
train_pk = x_train['pk']
valid_pk = x_valid['pk']
test_pk = X_test['pk']
predict_pk = needPredict['pk']


# Change negative values to positive for log transformation, square root transformation
train_data = x_train.join(y_train)
train_data['floor'] = train_data['floor']+2
train_data['school_1km'] = train_data['school_1km']+2
needPredict['floor'] = needPredict['floor']+2
needPredict['school_1km'] = needPredict['school_1km']+2


# Log transformation, square root transformation used for normal distribution
train_data[['school_1km', 'area']] = np.sqrt(train_data[['school_1km', 'area']])
train_data[['actual','floor', 'official_price', 'closest_sub', 'closest_high']] = np.log(train_data[['actual','floor', 'official_price', 'closest_sub', 'closest_high']])
needPredict[['school_1km', 'area']] = np.sqrt(needPredict[['school_1km', 'area']])
needPredict[['floor', 'official_price', 'closest_sub', 'closest_high']] = np.log(needPredict[['floor', 'official_price', 'closest_sub', 'closest_high']])


# Drop non-numerical data for ML
x_train, y_train = train_data.drop(['actual', 'pk', '시도'], axis=1), train_data['actual']
realX = needPredict.drop(['actual', 'pk', '시도'], axis=1)



# Same process for valid dataset
valid_data = x_valid.join(y_valid)

valid_data['floor'] = valid_data['floor']+2
valid_data['school_1km'] = valid_data['school_1km']+2

valid_data[['school_1km', 'area']] = np.sqrt(valid_data[['school_1km', 'area']])
valid_data[['actual','floor', 'official_price', 'closest_sub', 'closest_high']] = np.log(valid_data[['actual','floor', 'official_price', 'closest_sub', 'closest_high']])

x_valid, y_valid = valid_data.drop(['actual', 'pk', '시도'], axis=1), valid_data['actual']



# Same process for test dataset
test_data = X_test.join(y_test)

test_data['floor'] = test_data['floor']+2
test_data['school_1km'] = test_data['school_1km']+2

test_data[['school_1km', 'area']] = np.sqrt(test_data[['school_1km', 'area']])
test_data[['actual_price','floor', 'official_price', 'closest_sub', 'closest_high']] = np.log(test_data[['actual','floor', 'official_price', 'closest_sub', 'closest_high']])

y_test_actual_price = test_data['actual']
X_test, y_test = test_data.drop(['actual', 'actual_price', 'pk', '시도'], axis=1), test_data['actual_price']



# Normalization
scaler = MinMaxScaler()

scaler.fit(x_train)
x_train_s = scaler.transform(x_train)
x_valid_s = scaler.transform(x_valid)
X_test_s = scaler.transform(X_test)



# Running Machine Learning Model - Decision Tree Regression

from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import r2_score

tree = DecisionTreeRegressor()
tree.fit(x_train_s, y_train)
print(' ')
print('====== Decision Tree Regression ======')
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



# Predicting Unseen Prices using the same model

needPredict_s = scaler.transform(realX)
finalPred = tree.predict(needPredict_s)
needPred_df = pd.DataFrame(finalPred, columns=['예측가격'])
needPred_df['pk'] = predict_pk.values
needPred_df['predicted_price'] = np.exp(needPred_df['예측가격'])
needPred_df.drop('예측가격', axis=1, inplace=True)
print(' ')
print('<< null 예측값 >>')
print(needPred_df)



# Reverting Test Dataset Predicted Price values to real scale

pred_tr = pd.DataFrame(final_pred_tree, columns=['예측가격'])
pred_tr['pk'] = test_pk.values
pred_tr['predicted_price'] = np.exp(pred_tr['예측가격'])
pred_tr['actual_price'] = y_test_actual_price.values
pred_tr['predicted_price'] = pred_tr['predicted_price'].astype(float)

pred_tr.drop('예측가격', axis=1, inplace=True)


# Visualizing test dataset Error Distribution
pred_tr['error'] = (pred_tr['actual_price'] - pred_tr['predicted_price']) / pred_tr['actual_price'] * 100
print(' ')
print(pred_tr.describe())
pred_tr['error'].hist(bins=50)
plt.rc("axes", unicode_minus=False)
plt.rc('font', family='NanumGothic')
plt.title('Error Distribution in Decision Tree Regression Model : Daejeon')
plt.show()

test_df = pred_tr[['pk', 'predicted_price']]
total_df = pd.concat([needPred_df, test_df])

# Visualizing test dataset 'Predicted Price vs. Actual Price'
final_tr = pred_tr[['predicted_price', 'actual_price']]

plt.rc('font', family='NanumGothic')
final_tr.iloc[1700:1800, :].plot(figsize=(30,7))
plt.show()



# Storing data into MySQL table
engine = create_engine("mysql+pymysql://jiwoo:1234@localhost:3306/TESTDB?charset=utf8mb4")
conn = engine.connect()
table_name = 'predicted_price'
start_time = time.time()
total_df.to_sql(name=table_name, con=engine, if_exists='append', index=False)
end_time = time.time()
taketime = str(end_time - start_time)
print(table_name+" 스크립트 실행 소요시간 (단위 초) : " + taketime)
print(' ')
print('<< test set & null values prediction >>')
print(total_df)
    

# Showing the dataframe
pd.set_option('display.max_columns', None)
print(' ')
print("======== Test Dataset 'Predicted Price vs. Actual Price' =========")
print(final_tr)

