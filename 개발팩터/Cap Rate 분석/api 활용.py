import pandas as pd
import numpy as np
from urllib.parse import urlencode, quote_plus
from urllib.request import urlopen
import requests
from bs4 import BeautifulSoup
import bs4

code = pd.read_table('/home/subin/다운로드/팩터 논문/법정동코드 전체자료.txt', encoding='cp949')
code['시군구코드'] = code['법정동코드'].astype(str).str[0:5]
code['읍면동코드'] = code['법정동코드'].astype(str).str[5:10]
code['시도'] = code['법정동명'].str.split().str[0]
code['시군구'] = code['법정동명'].str.split().str[1]
code['읍면동'] = code['법정동명'].str.split().str[2]
code = code[['시군구코드', '읍면동코드', '시도', '시군구', '읍면동']].drop_duplicates()

gu_code_list = code['시군구코드'].unique().tolist()

for_leftover = code.iloc[1112:, :]
leftover_list = for_leftover['시군구코드'].unique().tolist()

url ="http://openapi.molit.go.kr:8081/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptRent?"
service_key = "서비스키"

base_date = ["202202", "202203", "202204"]

rowList = []

for i in range(len(base_date)):
    for l in range(len(gu_code_list)):

        #gu_code = gu_code_list[l]       # 시군구코드를 모두 담고 있는 gu_code_list에서 순차적으로 불러오기
        gu_code = gu_code_list[l]

        payload = "serviceKey=" + service_key + "&"+"LAWD_CD=" + gu_code + "&"+"DEAL_YMD=" + base_date[i]+ "&" 
        res = requests.get(url + payload).text
        xmlobj = bs4.BeautifulSoup(res, 'lxml-xml')
        rows = xmlobj.findAll('item')       # 하나의 시군구에 대하여 여러 row에 해당하는 데이터 row가 아닌 다른 형태로 불려옴.

        columnList = []        

        rowsLen = len(rows)         # row에 해당하는 데이터 개수

        for k in range(0, rowsLen):                     # rowList에 하나의 시군구, 하나의 시점에 해당하는 모든 row 저장
            columns = rows[k].find_all()
        
            columnsLen = len(columns)       # 데이터프레임 상에서 column으로 들어가야 할 데이터를 하나의 row에서 추출하여 정리
        
            for j in range(0, columnsLen):         
                eachColumn = columns[j].text
                columnList.append(eachColumn)       
            rowList.append(columnList)      # 한 개의 row를 rowList에 append해서 저장
            columnList = []  
result___ = pd.DataFrame(rowList)
    
result___.to_csv('file_name_020304.csv')
