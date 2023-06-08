import pandas as pd
from sqlalchemy import create_engine
import time

df = pd.read_csv('/home/subin/여피 5월/전국-학교merge.csv')

sidolist = df['시도'].unique().tolist()

k=0
while k<len(sidolist):
    engine = create_engine("mysql+pymysql://jiwoo:1234@localhost:3306/TESTDB?charset=utf8mb4")
    conn = engine.connect()
    table_name = sidolist[k]+'공시지가만_preprocessing'
    start_time = time.time()
    part = df[df['시도']==sidolist[k]]
    part.to_sql(name=table_name, con=engine, if_exists='append', index=False)
    end_time = time.time()
    taketime = str(end_time - start_time)
    print(table_name+" 스크립트 실행 소요시간 (단위 초) : " + taketime)
    k=k+1
print('============= 끝 ==============')