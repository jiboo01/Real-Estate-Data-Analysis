# Price Prediction for Apartments in Korea
### Project Aim
The Korean Government has made available housing transaction prices and appraised prices data for properties in Korea. However, not all properties have data for both transaction prices and appraised prices. Some properties only have information on the appraised price. Consequently, this project aims to estimate the transaction prices for apartments lacking such data, utilizing various variables including appraised prices, distances to the nearest subway station and high school, and other relevant factors.

### Areas for improvement
1) More factors to be added

Commercial facilities, job opportunities, NIMBY(not in my backyard) etc.

2) Incomplete floor data

The 'floor' column was determined by extracting the initial numeric value from the 'ho' column. Consequently, uncertain values were uniformly assigned as 1, representing the first floor, which may not accurately reflect the actual floors.

3) Actual distance missing

The 'distance' columns indicate the direct distances between two points, such as the apartment-school or apartment-subway station. However, these distances do not account for the actual traveling distance, which may differ due to various factors such as road layouts, traffic conditions, or available transportation routes.

---   

### Files
* Cleaning Data : [3. Price Prediction (ML Model)/수도권 실거래가 예측 모델_수정.ipynb](https://github.com/jiboo01/hsj/blob/main/3.%20Price%20Prediction%20(ML%20Model)/%EC%88%98%EB%8F%84%EA%B6%8C%20%EC%8B%A4%EA%B1%B0%EB%9E%98%EA%B0%80%20%EC%98%88%EC%B8%A1%20%EB%AA%A8%EB%8D%B8_%EC%88%98%EC%A0%95.ipynb)

    In [4] ~ In [7] : Mapping Appraised Price & Address Data

    In [13] ~ In [46] : Creating 'floor' column by cleaning 'ho' column

    In [51] ~ In [59] : Cleaning Address Data for mapping Geospatial Data

        전국 모든 지역에 대한 주소 데이터 cleaning은 아래 '수도권_동별 Clustering 공시지가 copy.ipynb' 참고

   [수도권_동별 Clustering 공시지가 copy.ipynb](https://github.com/jiboo01/hsj/blob/main/%EC%88%98%EB%8F%84%EA%B6%8C_%EB%8F%99%EB%B3%84%20Clustering%20%EA%B3%B5%EC%8B%9C%EC%A7%80%EA%B0%80%20copy.ipynb) : In [10] ~ In [22]

    In [61] ~ In [85] : Calculating direct distance to School & Subway st.

    In [88] ~ In [95] : Mapping Transaction Price Data to Appraised Price Data

        머신러닝 코드는 아래 'Final ML Model'을 사용할 것

  ~~In [96] ~ In [112] : Preprocessing Data for ML~~

  ~~In [113] ~ In [125] : Testing various ML Models~~

  ~~In [126] ~ : Visualizing Accuracy of prediction~~


  
* Final ML Model : [3. Price Prediction (ML Model)/ML모델 DB넣기.py](https://github.com/jiboo01/hsj/blob/main/3.%20Price%20Prediction%20(ML%20Model)/ML%EB%AA%A8%EB%8D%B8%20DB%EB%84%A3%EA%B8%B0.py)

    Line 13 ~ 44 : Loading cleaned data from MySQL and turning it into a dataframe

    Line 47 ~ 70 : Splitting Data for ML

    Line 73 ~ 127 : Preprocessing Data for ML

    Line 131 ~ 181 : Running & Testing ML Model

    Line 185 ~ 228 : Visualizing the Results of Price Prediction

    Line 232 ~ 243 : Storing Result into MySQL
  
* Further Analysis : [3. Price Prediction (ML Model)/분석](https://github.com/jiboo01/hsj/tree/main/3.%20Price%20Prediction%20(ML%20Model)/%EB%B6%84%EC%84%9D)

* Data Used : [data](https://github.com/jiboo01/hsj/tree/main/data)
### Requirements
absl-py @ file:///home/conda/feedstock_root/build_artifacts/absl-py_1673535674859/work
aiohttp @ file:///home/conda/feedstock_root/build_artifacts/aiohttp_1649013149994/work
aiosignal @ file:///home/conda/feedstock_root/build_artifacts/aiosignal_1667935791922/work
asgiref==3.7.2
astunparse @ file:///home/conda/feedstock_root/build_artifacts/astunparse_1610696312422/work
async-timeout @ file:///home/conda/feedstock_root/build_artifacts/async-timeout_1640026696943/work
attrs @ file:///home/conda/feedstock_root/build_artifacts/attrs_1671632566681/work
blinker @ file:///home/conda/feedstock_root/build_artifacts/blinker_1664823096650/work
Bottleneck @ file:///opt/conda/conda-bld/bottleneck_1657175564434/work
brotlipy==0.7.0
cachetools @ file:///home/conda/feedstock_root/build_artifacts/cachetools_1674482203741/work
certifi @ file:///croot/certifi_1671487769961/work/certifi
cffi @ file:///croot/cffi_1670423208954/work
charset-normalizer @ file:///tmp/build/80754af9/charset-normalizer_1630003229654/work
click @ file:///home/conda/feedstock_root/build_artifacts/click_1666798198223/work
conda==23.1.0
conda-content-trust @ file:///tmp/abs_5952f1c8-355c-4855-ad2e-538535021ba5h26t22e5/croots/recipe/conda-content-trust_1658126371814/work
conda-package-handling @ file:///croot/conda-package-handling_1672865015732/work
conda_package_streaming @ file:///croot/conda-package-streaming_1670508151586/work
contourpy==1.0.7
cryptography @ file:///croot/cryptography_1673298753778/work
cycler==0.11.0
Django==4.2.2
et-xmlfile==1.1.0
flatbuffers==23.3.3
fonttools==4.39.0
frozenlist @ file:///croot/frozenlist_1670004507010/work
gast @ file:///home/conda/feedstock_root/build_artifacts/gast_1596839682936/work
google-auth @ file:///home/conda/feedstock_root/build_artifacts/google-auth_1679641775083/work
google-auth-oauthlib @ file:///home/conda/feedstock_root/build_artifacts/google-auth-oauthlib_1630497468950/work
google-pasta==0.2.0
grpcio==1.53.0
h5py==3.8.0
idna @ file:///croot/idna_1666125576474/work
importlib-metadata @ file:///home/conda/feedstock_root/build_artifacts/importlib-metadata_1679167925176/work
jax==0.4.7
keras==2.12.0
Keras-Preprocessing @ file:///home/conda/feedstock_root/build_artifacts/keras-preprocessing_1610713559828/work
kiwisolver==1.4.4
libclang==16.0.0
Markdown @ file:///home/conda/feedstock_root/build_artifacts/markdown_1679584000376/work
MarkupSafe==2.1.2
matplotlib==3.7.1
ml-dtypes==0.0.4
multidict @ file:///croot/multidict_1665674239670/work
numexpr @ file:///croot/numexpr_1668713893690/work
numpy==1.24.2
oauthlib @ file:///home/conda/feedstock_root/build_artifacts/oauthlib_1666056362788/work
openpyxl==3.1.2
opt-einsum @ file:///home/conda/feedstock_root/build_artifacts/opt_einsum_1617859230218/work
packaging @ file:///home/conda/feedstock_root/build_artifacts/packaging_1673482170163/work
pandas==1.5.3
Pillow==9.4.0
pluggy @ file:///tmp/build/80754af9/pluggy_1648024709248/work
protobuf==3.20.3
pyarrow==11.0.0
pyasn1==0.4.8
pyasn1-modules==0.2.8
pycosat @ file:///croot/pycosat_1666805502580/work
pycparser @ file:///tmp/build/80754af9/pycparser_1636541352034/work
PyJWT @ file:///home/conda/feedstock_root/build_artifacts/pyjwt_1666240235902/work
pyOpenSSL @ file:///opt/conda/conda-bld/pyopenssl_1643788558760/work
pyparsing==3.0.9
PySocks @ file:///home/builder/ci_310/pysocks_1640793678128/work
python-dateutil @ file:///tmp/build/80754af9/python-dateutil_1626374649649/work
pytz==2022.7.1
pyu2f @ file:///home/conda/feedstock_root/build_artifacts/pyu2f_1604248910016/work
requests @ file:///opt/conda/conda-bld/requests_1657734628632/work
requests-oauthlib @ file:///home/conda/feedstock_root/build_artifacts/requests-oauthlib_1643557462909/work
rsa @ file:///home/conda/feedstock_root/build_artifacts/rsa_1658328885051/work
ruamel.yaml @ file:///croot/ruamel.yaml_1666304550667/work
ruamel.yaml.clib @ file:///croot/ruamel.yaml.clib_1666302247304/work
scipy==1.10.1
seaborn==0.12.2
six @ file:///tmp/build/80754af9/six_1644875935023/work
sqlparse==0.4.4
tensorboard==2.12.0
tensorboard-data-server==0.7.0
tensorboard-plugin-wit @ file:///home/conda/feedstock_root/build_artifacts/tensorboard-plugin-wit_1641458951060/work/tensorboard_plugin_wit-1.8.1-py3-none-any.whl
tensorflow==2.12.0
tensorflow-estimator==2.12.0
tensorflow-io-gcs-filesystem==0.31.0
termcolor @ file:///home/conda/feedstock_root/build_artifacts/termcolor_1672833821273/work
toolz @ file:///croot/toolz_1667464077321/work
tqdm @ file:///opt/conda/conda-bld/tqdm_1664392687731/work
typing_extensions @ file:///home/conda/feedstock_root/build_artifacts/typing_extensions_1678559861143/work
urllib3 @ file:///croot/urllib3_1673575502006/work
Werkzeug==2.2.3
wrapt @ file:///tmp/abs_c335821b-6e43-4504-9816-b1a52d3d3e1eel6uae8l/croots/recipe/wrapt_1657814400492/work
xlrd==2.0.1
yarl @ file:///home/conda/feedstock_root/build_artifacts/yarl_1648966516552/work
zipp @ file:///home/conda/feedstock_root/build_artifacts/zipp_1677313463193/work
zstandard @ file:///opt/conda/conda-bld/zstandard_1663827383994/work

