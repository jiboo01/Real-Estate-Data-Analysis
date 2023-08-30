# data_processing
### 폴더 설명
1. data : 필요한 데이터 파일
2. Cap Rate : Cap Rate 계산 과정
3. 상업용 기준시가 : 상업용 기준시가 계산 과정
### Cap Rate 파일 활용 순서
1) requirements.txt 설치
2) api 활용.py : 공공데이터포털에서 전월세데이터 불러온 뒤 .csv로 저장 (일일트래픽 1000 유의, 하나의 서비스키로 약 2.5개월치 데이터 불러오기 가능)
3) Cap Rate 2022년 전국 full process.ipynb : 데이터 전처리, 맵핑, Cap Rate Table 생성, 분석 및 시각화 과정 모두 상세히 포함
4) Cap Rate 2022년 전국 code only.py : 데이터 전처리, 맵핑, Cap Rate Table 생성 코드만 포함
