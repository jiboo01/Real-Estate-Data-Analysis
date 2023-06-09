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
sql = "SELECT pk, area, floor FROM daejeon_area_floor"
cur.execute(sql)
rows = cur.fetchall()
con.close()

area_floor = pd.DataFrame(rows)
end_time = time.time()
taketime = str(end_time - start_time)
#print("import 실행 소요시간 (단위 초) : " + taketime)
#print(area_floor)

start_time = time.time()
con = pymysql.connect(host='localhost', user='jiwoo', password='1234', db='TESTDB', charset='utf8', autocommit=True, cursorclass=pymysql.cursors.DictCursor)
cur = con.cursor()
sql = "SELECT * FROM `대전광역시공시지가만 preprocessing`"
cur.execute(sql)
rows = cur.fetchall()
con.close()

total = pd.DataFrame(rows)
end_time = time.time()
taketime = str(end_time - start_time)
#print("import 실행 소요시간 (단위 초) : " + taketime)

#print(total)

df = total.merge(area_floor, on='pk', how='left')
pd.set_option('display.max_columns', None)
print(df.dtypes)

df1 = pd.read_csv('/home/subin/다운로드/여피/0308 업무-20230310T030110Z-001/0308 업무/complex_20230308 (사본).csv', header=None)
df1.columns = ['old_addr_idx', 'complex_name', 'property_type','sido', '시군구',
               '읍면동','리','road_name','road_code','admin_dong_code','postal_code','jibun_main','jibun_sub','updated']
df1.drop(['road_name','road_code','admin_dong_code','postal_code','jibun_main','jibun_sub', 'updated'], axis=1, inplace=True)

df1 = df1[df1['sido']=='대전광역시']
df1 = df1[df1['property_type']!='officetel']
df1 = df1[['old_addr_idx', 'sido']]
df1 = df1.drop_duplicates()
print(df1['old_addr_idx'].nunique())
print(df1.dtypes)
pd.set_option('display.max_columns', None)
print(df1)

df3 = pd.read_csv('/home/subin/다운로드/여피/0315 업무/trade_history_2018-2023_edit.csv', header=None)
df3.columns = ['old_addr_idx', '거래날짜', 'actual', '면적', 'floor', '모름', '거래방식', '모름2', '모름3', '시군구', '주소', '읍면동', 'complex_name', '모름4', '모름5', 'property_type']
df3 = df3.drop_duplicates() 
df3['area'] = df3['면적']/3.3
df3.fillna("NONE", inplace=True)
df3 = df3[~df3['property_type'].str.contains('officetel')]
df3 = df3[['old_addr_idx', 'actual', 'area', 'floor', '거래날짜']]
df3 = df3.drop_duplicates()
df3['area'] = df3['area'].astype(float).round(0)
df3['area'] = df3['area'].astype(int)
df3 = df3.merge(df1, on='old_addr_idx', how='left')
print(df3['old_addr_idx'].nunique())
print(df3.dtypes)
print(df3)
real_trainInfo = df.merge(df3, on=['old_addr_idx', 'area', 'floor'], how='left')
real_trainInfo = real_trainInfo.dropna(subset='sido')
real_trainInfo = real_trainInfo[real_trainInfo['시도'].isnull()==True]
real_trainInfo = real_trainInfo.drop(['subway_id', 'school_id'], axis=1)

print(real_trainInfo)
print(real_trainInfo.isnull().sum())
print(df)

engine = create_engine("mysql+pymysql://jiwoo:1234@localhost:3306/TESTDB?charset=utf8mb4")
conn = engine.connect()
table_name = 'off_act_price'
start_time = time.time()
real_trainInfo.to_sql(name=table_name, con=engine, if_exists='append', index=False)
end_time = time.time()
taketime = str(end_time - start_time)
print(table_name+" 스크립트 실행 소요시간 (단위 초) : " + taketime)