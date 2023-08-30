# data_processing
### 폴더 설명
1. data : 필요한 데이터 파일
2. Cap Rate : Cap Rate 계산 과정
3. 상업용 기준시가 : 상업용 기준시가 계산 과정

---   

### Files
* [requirements.txt](https://github.com/hansangjik-and-company/data_processing/blob/main/requirements.txt) : 설치 패키지
  
* [data](https://github.com/hansangjik-and-company/data_processing/tree/main/data) : 사용된 데이터 파일

    * 파일이 안 불려오는 경우, 파일 경로 수정 필요
    * 실거래가 & 공시지가 데이터는 용량이 너무 커서 업로드 불가 ∴직접 파일 생성 or DB에서 불러오는 등의 작업 필요

* [api 활용.py](https://github.com/hansangjik-and-company/data_processing/blob/main/Cap_Rate/api%20%ED%99%9C%EC%9A%A9.py) : 공공데이터 오픈API 활용방법
  
* [Cap Rate/Cap Rate 2022년 전국 code only.py](https://github.com/hansangjik-and-company/data_processing/blob/main/Cap_Rate/Cap%20Rate%202022%EB%85%84%20%EC%A0%84%EA%B5%AD%20code%20only.py) : 데이터 전처리, 맵핑, Cap Rate Table 생성 코드만 포함 

    Line 5 ~ 123 : '법정동코드 전체자료.txt' 데이터 전처리

    Line 125 ~ 184 : 전월세 데이터 전처리

    Line 186 ~ 208 : 법정동코드 & 전월세데이터 맵핑 및 추가 전처리 및 Cap Rate 순수익 계산을 위한 컬럼 생성

    Line 210 ~ 217 : 실거래가와 맵핑할 최종 전월세데이터 테이블 생성

    Line 219 ~ 310 : 공시지가 데이터를 활용하여 실거래가 데이터의 주소 데이터 전처리 (전월세데이터와 맵핑 가능하도록 하기 위함)

    Line 312 ~ 378 : 전월세 데이터와 맵핑할 최종 실거래가 데이터 테이블 생성 (+추가 전처리 및 Cap Rate 현재가치 컬럼 생성)

    Line 380 ~ 404 : Cap Rate 계산 과정 및 최종 Cap Rate Table 생성
  
* [Cap Rate/Cap Rate 2022년 전국 full process.ipynb](https://github.com/hansangjik-and-company/data_processing/blob/main/Cap_Rate/Cap%20Rate%202022%EB%85%84%20%EC%A0%84%EA%B5%AD%20full%20process.ipynb) : 데이터 전처리, 맵핑, Cap Rate Table 생성, 분석 및 시각화 과정 모두 상세히 포함

    In [2] : '법정동코드 전체자료.txt' 데이터 전처리

    In [3] ~ Out [6] : 전월세 데이터 전처리

    In [7]: 법정동코드 & 전월세데이터 맵핑

    In [8] ~ Out [10] : 맵핑 후 테이블 전처리 및 Cap Rate 순수익 계산을 위한 컬럼 생성

    In [11] ~ In [12] : 실거래가와 맵핑할 최종 전월세데이터 테이블 생성

    In [14] ~ Out [16] : 전월세데이터 분석 및 시각화

    In [17] ~ Out [18] : 실거래가 & 공시지가 데이터 불러오기

    In [19] ~ Out [23] : 공시지가 데이터를 활용하여 실거래가 데이터의 주소 데이터 전처리 (전월세데이터와 맵핑 가능하도록 하기 위함)

    ~~In [24] ~ Out [26] : 추후 데이터 전처리가 필요한 부분~~

    In [29] ~ Out [31] : 전월세 데이터와 맵핑할 최종 실거래가 데이터 테이블 생성 (+추가 전처리 및 Cap Rate 현재가치 컬럼 생성)

    In [33] ~ Out [33] : 전월세 데이터 & 실거래가 데이터 맵핑 -> Cap Rate Table 생성

    In [34] ~ In [36] : Cap Rate Table 데이터 분석 및 시각화

    In [37] ~ Out [37] : Cap Rate 계산 과정

    In [38] ~ Out [46] : Cap Rate Table 추가 데이터 분석 및 시각화
