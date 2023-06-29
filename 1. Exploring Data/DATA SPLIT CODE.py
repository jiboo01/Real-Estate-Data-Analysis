import pandas as pd

## 시군구 정보가 제대로 들어가있지 않아 data cleaning 진행했었음.
## 과정이 훨씬 긴데 여기엔 일부만 나타나있음.

df1 = pd.read_csv('./complex_20230308.csv', header=None)
df1.columns = ['old_addr_idx', 'complex_name', 'property_type','시도','시군구',
               '읍면동','리','road_name','road_code','admin_dong_code','postal_code','jibun_main','jibun_sub','updated']

df2 = pd.read_csv('./property_20230308.csv', header=None)
df2.columns = ['pk','old_addr_idx','dong','ho','official_price','net_leasable_area','updated']

df_tot = df2.merge(df1, on='old_addr_idx')

df_apt_tot = df_tot[df_tot['property_type']=='apartment']

pd.options.display.max_columns = None
df_apt_tot['평수'] = df_apt_tot['net_leasable_area'] / 3.3

ranges = [(0, 10, '10평 미만'), (10, 20, '10평대'), (20, 30, '20평대'), (30, 40, '30평대'), (40,50,'40평대'), (50,60,'50평대'),(60,70,'60평대'),(70,80,'70평대'),(80,90,'80평대'),(90,170,'90평대 이상')]

labels = [r[2] for r in ranges]
bins = [r[0] for r in ranges] + [ranges[-1][1]]

df_apt_tot['평형'] = pd.cut(df_apt_tot['평수'], bins=bins, labels=labels, include_lowest=True, right=False)


df_apt_tot['시군구_split'] = df_apt_tot['시군구'].str.split()
split_list = df_apt_tot['시군구_split'].apply(lambda x: x[1:] if (type(x) == list and len(x) > 1) else x)
df_apt_tot['시군구_최종'] = split_list
df_apt_gyeongghi = df_apt_tot[df_apt_tot['시도']=='경기도']
print(df_apt_gyeongghi)