#!/usr/bin/env python
# coding: utf-8

# In[1]:


# import libraries and read .rds as a pandas dataframe

import os
import pandas as pd
import pyreadr
import math
from operator import itemgetter
import statistics
import time
import matplotlib.pyplot as plt

os.getcwd()
os.chdir('/Users/jicheolha/Downloads')


# In[2]:


# original dataframe

df = pyreadr.read_r('gq_with_coord_2_industries.rds')[None].dropna()
df = df[df.index % 1 == 0]


# In[3]:


# dataframe with GPT calculations

df2 = pyreadr.read_r('gq_17_28_distance_province.rds')[None].dropna()


# In[4]:


# select for 1998 Beijing only

df2[(df2['province_code'] == 110000) & (df2['year'] == 1998)]


# In[5]:


# define haversine distance function

def haversine_distance(point1, point2):
    lat1, lon1 = point1
    lat2, lon2 = point2
    
    # Convert degrees to radians
    lat1 = math.radians(lat1)
    lon1 = math.radians(lon1)
    lat2 = math.radians(lat2)
    lon2 = math.radians(lon2)
    
    # Haversine formula
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    
    # Earth's radius in kilometers
    R = 6371.0
    
    # Calculate the distance
    distance = R * c
    
    return distance


# In[6]:


# find n1 and n2 and the number of unique years, provinces, cities, and counties

n1 = df['industry_code'].value_counts()[17]
n2 = df['industry_code'].value_counts()[28]

n_yr = len(df['year'].unique())
n_pc = len(df['province_code'].unique())
n_cic = len(df['city_code'].unique())
n_coc = len(df['county_code'].unique())

# print(n1,n2)
# print(n_yr, n_pc, n_cic, n_coc)


# In[7]:


# divide the dataframe into two according to industry types

df17 = df[df['industry_code'] == 17][['year','firm_id','industry_code','province_code','city_code','county_code','latitude','longitude']]
df28 = df[df['industry_code'] == 28][['year','firm_id','industry_code','province_code','city_code','county_code','latitude','longitude']]


# In[8]:


# sort by year/province, year/city, year/county
# create lists of ordered year, province, city, and county

grpd_yr_pr = sorted(df17.values.tolist(), key = itemgetter(0,3))
grpd_yr_cic = sorted(df17.values.tolist(), key = itemgetter(0,4))
grpd_yr_coc = sorted(df17.values.tolist(), key = itemgetter(0,5))

yr = list(set(map(itemgetter(0),grpd_yr_pr)))[0:1]
pr = list(set(map(itemgetter(3),grpd_yr_pr)))[9:10]
cic = list(set(map(itemgetter(4),grpd_yr_cic)))
coc = list(set(map(itemgetter(5),grpd_yr_coc)))

tuple_ls = [(year, province_code, df17.loc[(df17['year'] == year) & (df17['province_code'] == province_code)]) for year in yr for province_code in pr]
triple_ls = [(year, province_code, df28.loc[(df28['year'] == year) & (df28['province_code'] == province_code)]) for year in yr for province_code in pr]


# In[9]:


# average distance within for industry 17 by each year for each province
start = time.time()
col1 = []
distance = 0
for (year, province_code, df) in tuple_ls:
    for k in range(len(df)): 
        distance = statistics.mean(haversine_distance(df[['latitude','longitude']].iloc[k],df[['latitude','longitude']].iloc[l]) for l in range(len(df)))
        col1.append(distance)
    distance = 0
    
print(col1)

end = time.time()
print(end - start)


# In[10]:


# average distance within for industry 28 by each year for each province

start = time.time()
col2 = []
distance = 0
for (year, province_code, df) in triple_ls:
    for k in range(len(df)): 
        distance = statistics.mean(haversine_distance(df[['latitude','longitude']].iloc[k],df[['latitude','longitude']].iloc[l]) for l in range(len(df)))
        col2.append(distance)
    distance = 0

print(col2)

end = time.time()
print(end - start)


# In[11]:


# print(len(col1),len(col2))


# In[12]:


# turn results into pandas dataframe

output = pd.DataFrame(col1+col2)
output.columns=['code_within']


# In[13]:


# append the results to 1998 Beijing dataframe and calculate differences

df_final = pd.concat([df2[(df2['province_code'] == 110000) & (df2['year'] == 1998)][['year','firm_id', 'province_code','industry_code','within_distance']]
,output], axis = 1)
df_final['within_distance'] = df_final['within_distance']/1000
df_final['abs difference from GPT'] = abs(df_final['within_distance']-df_final['code_within'])
df_final['difference from GPT (%)'] = abs(df_final['within_distance']-df_final['code_within'])/df_final['within_distance']
df_final


# In[14]:


# save result as .csv

df_final.to_csv('df_final.csv', index=False)


# In[15]:


# plot the absolute difference and save as .pdf

plt.bar(df_final.index, df_final['abs difference from GPT'])
plt.xlabel('Firm ID')
plt.ylabel('Distance (km)')
plt.title('Absolute Difference Between GPT and Code')
plt.savefig('absolute_difference.pdf', format='pdf')
plt.show()


# In[16]:


# plot the percentage difference and save as .pdf

plt.bar(df_final.index, df_final['difference from GPT (%)'])
plt.xlabel('Firm ID')
plt.ylabel('%')
plt.title('Percentage Difference Between GPT and Code')
plt.ylim(0,0.1)
plt.savefig('percent_difference.pdf', format='pdf')
plt.show()


# In[17]:


# another method?

def calculate_avg_distance(group_df):
    # Calculate the mean of pairwise differences (location - other_location)
    pairwise_diffs = []
    col = []
    for _, value1 in group_df[['longitude', 'latitude']].iterrows():
        longitude_1 = list(value1.items())[0][1]
        latitude_1 = list(value1.items())[1][1]
        for _, value2 in group_df[['longitude', 'latitude']].iterrows():
            longitude_2 = list(value2.items())[0][1]
            latitude_2 = list(value2.items())[1][1]
            dist = haversine_distance((latitude_1,longitude_1),(latitude_2,longitude_2))
            pairwise_diffs.append(dist)
    if pairwise_diffs:
        col.append(np.mean(pairwise_diffs))
    else:
        col.append(np.nan)  # Return NaN if there's only one record in the group

