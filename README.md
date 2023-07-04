# Price Prediction for Apartments in Korea
### Project Aim
The Korean Government has made available housing transaction prices and appraised prices data for properties in Korea. However, not all properties have data for both transaction prices and appraised prices. Some properties only have information on the appraised price. Consequently, this project aims to estimate the transaction prices for apartments lacking such data, utilizing various variables including appraised prices, distances to the nearest subway station and high school, and other relevant factors.
### Data used

### Areas for improvement
1) More factors to be added

Commercial facilities, job opportunities, NIMBY(not in my backyard) etc.
2) Incomplate floor data

The 'floor' column was determined by extracting the initial numeric value from the 'ho' column. Consequently, uncertain values were uniformly assigned as 1, representing the first floor, which may not accurately reflect the actual floors.
3) Actual distance missing

The 'distance' columns indicate the direct distances between two points, such as the apartment-school or apartment-subway station. However, these distances do not account for the actual traveling distance, which may differ due to various factors such as road layouts, traffic conditions, or available transportation routes.
---   
### Files
[For full process (From preprocessing to ML & visualization)](https://github.com/jiboo01/hsj/blob/main/3.%20Price%20Prediction%20(ML%20Model)/%EC%88%98%EB%8F%84%EA%B6%8C%20%EC%8B%A4%EA%B1%B0%EB%9E%98%EA%B0%80%20%EC%98%88%EC%B8%A1%20%EB%AA%A8%EB%8D%B8_%EC%88%98%EC%A0%95.ipynb)
[Import & Export ML outputs](https://github.com/jiboo01/hsj/blob/main/3.%20Price%20Prediction%20(ML%20Model)/ML%EB%AA%A8%EB%8D%B8%20DB%EB%84%A3%EA%B8%B0.py)
[Further Analysis](https://github.com/jiboo01/hsj/tree/main/3.%20Price%20Prediction%20(ML%20Model)/%EB%B6%84%EC%84%9D)
