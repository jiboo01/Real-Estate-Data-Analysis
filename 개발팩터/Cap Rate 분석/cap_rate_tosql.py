import pymysql
from sqlalchemy import create_engine
import time
import pandas as pd

cap_rate_2022 = pd.read_csv('/home/subin/다운로드/팩터 논문/cap_rate_2022.csv')
cap_rate_2022 = cap_rate_2022.drop('Unnamed: 0', axis=1)
engine = create_engine("mysql+pymysql://jiwoo:" + 비밀번호 + "@localhost:3306/yuppie?charset=utf8mb4")
conn = engine.connect()
table_name = 'cap_rate_2022'
start_time = time.time()
cap_rate_2022.to_sql(name=table_name, con=engine, if_exists='append', index=False)
end_time = time.time()
taketime = str(end_time - start_time)
print(table_name+" 스크립트 실행 소요시간 (단위 초) : " + taketime)