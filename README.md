```python
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.diagnostic import linear_rainbow, het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

from sklearn.preprocessing import LabelEncoder
```

# Inferential Modeling Workflow

This dataset was downloaded from [Kaggle](https://www.kaggle.com/kumarajarshi/life-expectancy-who) and reflects data collected by the WHO about life expectancy and potentially-related factors.  The information is aggregated on a per-country per-year basis.

The following questions have been posed:

1. Does various predicting factors which has been chosen initially really affect the Life expectancy? What are the predicting variables actually affecting the life expectancy?
2. Should a country having a lower life expectancy value(<65) increase its healthcare expenditure in order to improve its average lifespan?
3. How does Infant and Adult mortality rates affect life expectancy?
4. Does Life Expectancy has positive or negative correlation with eating habits, lifestyle, exercise, smoking, drinking alcohol etc.
5. What is the impact of schooling on the lifespan of humans?
6. Does Life Expectancy have positive or negative relationship with drinking alcohol?
7. Do densely populated countries tend to have lower life expectancy?
8. What is the impact of Immunization coverage on life Expectancy?

### Importing the Data


```python
df = pd.read_csv("data/life_expectancy.csv")
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Year</th>
      <th>Status</th>
      <th>Life expectancy</th>
      <th>Adult Mortality</th>
      <th>infant deaths</th>
      <th>Alcohol</th>
      <th>percentage expenditure</th>
      <th>Hepatitis B</th>
      <th>Measles</th>
      <th>...</th>
      <th>Polio</th>
      <th>Total expenditure</th>
      <th>Diphtheria</th>
      <th>HIV/AIDS</th>
      <th>GDP</th>
      <th>Population</th>
      <th>thinness  1-19 years</th>
      <th>thinness 5-9 years</th>
      <th>Income composition of resources</th>
      <th>Schooling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2015</td>
      <td>Developing</td>
      <td>65.0</td>
      <td>263.0</td>
      <td>62</td>
      <td>0.01</td>
      <td>71.279624</td>
      <td>65.0</td>
      <td>1154</td>
      <td>...</td>
      <td>6.0</td>
      <td>8.16</td>
      <td>65.0</td>
      <td>0.1</td>
      <td>584.259210</td>
      <td>33736494.0</td>
      <td>17.2</td>
      <td>17.3</td>
      <td>0.479</td>
      <td>10.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2014</td>
      <td>Developing</td>
      <td>59.9</td>
      <td>271.0</td>
      <td>64</td>
      <td>0.01</td>
      <td>73.523582</td>
      <td>62.0</td>
      <td>492</td>
      <td>...</td>
      <td>58.0</td>
      <td>8.18</td>
      <td>62.0</td>
      <td>0.1</td>
      <td>612.696514</td>
      <td>327582.0</td>
      <td>17.5</td>
      <td>17.5</td>
      <td>0.476</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2013</td>
      <td>Developing</td>
      <td>59.9</td>
      <td>268.0</td>
      <td>66</td>
      <td>0.01</td>
      <td>73.219243</td>
      <td>64.0</td>
      <td>430</td>
      <td>...</td>
      <td>62.0</td>
      <td>8.13</td>
      <td>64.0</td>
      <td>0.1</td>
      <td>631.744976</td>
      <td>31731688.0</td>
      <td>17.7</td>
      <td>17.7</td>
      <td>0.470</td>
      <td>9.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2012</td>
      <td>Developing</td>
      <td>59.5</td>
      <td>272.0</td>
      <td>69</td>
      <td>0.01</td>
      <td>78.184215</td>
      <td>67.0</td>
      <td>2787</td>
      <td>...</td>
      <td>67.0</td>
      <td>8.52</td>
      <td>67.0</td>
      <td>0.1</td>
      <td>669.959000</td>
      <td>3696958.0</td>
      <td>17.9</td>
      <td>18.0</td>
      <td>0.463</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2011</td>
      <td>Developing</td>
      <td>59.2</td>
      <td>275.0</td>
      <td>71</td>
      <td>0.01</td>
      <td>7.097109</td>
      <td>68.0</td>
      <td>3013</td>
      <td>...</td>
      <td>68.0</td>
      <td>7.87</td>
      <td>68.0</td>
      <td>0.1</td>
      <td>63.537231</td>
      <td>2978599.0</td>
      <td>18.2</td>
      <td>18.2</td>
      <td>0.454</td>
      <td>9.5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Year</th>
      <th>Life expectancy</th>
      <th>Adult Mortality</th>
      <th>infant deaths</th>
      <th>Alcohol</th>
      <th>percentage expenditure</th>
      <th>Hepatitis B</th>
      <th>Measles</th>
      <th>BMI</th>
      <th>under-five deaths</th>
      <th>Polio</th>
      <th>Total expenditure</th>
      <th>Diphtheria</th>
      <th>HIV/AIDS</th>
      <th>GDP</th>
      <th>Population</th>
      <th>thinness  1-19 years</th>
      <th>thinness 5-9 years</th>
      <th>Income composition of resources</th>
      <th>Schooling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2938.000000</td>
      <td>2928.000000</td>
      <td>2928.000000</td>
      <td>2938.000000</td>
      <td>2744.000000</td>
      <td>2938.000000</td>
      <td>2385.000000</td>
      <td>2938.000000</td>
      <td>2904.000000</td>
      <td>2938.000000</td>
      <td>2919.000000</td>
      <td>2712.00000</td>
      <td>2919.000000</td>
      <td>2938.000000</td>
      <td>2490.000000</td>
      <td>2.286000e+03</td>
      <td>2904.000000</td>
      <td>2904.000000</td>
      <td>2771.000000</td>
      <td>2775.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2007.518720</td>
      <td>69.224932</td>
      <td>164.796448</td>
      <td>30.303948</td>
      <td>4.602861</td>
      <td>738.251295</td>
      <td>80.940461</td>
      <td>2419.592240</td>
      <td>38.321247</td>
      <td>42.035739</td>
      <td>82.550188</td>
      <td>5.93819</td>
      <td>82.324084</td>
      <td>1.742103</td>
      <td>7483.158469</td>
      <td>1.275338e+07</td>
      <td>4.839704</td>
      <td>4.870317</td>
      <td>0.627551</td>
      <td>11.992793</td>
    </tr>
    <tr>
      <th>std</th>
      <td>4.613841</td>
      <td>9.523867</td>
      <td>124.292079</td>
      <td>117.926501</td>
      <td>4.052413</td>
      <td>1987.914858</td>
      <td>25.070016</td>
      <td>11467.272489</td>
      <td>20.044034</td>
      <td>160.445548</td>
      <td>23.428046</td>
      <td>2.49832</td>
      <td>23.716912</td>
      <td>5.077785</td>
      <td>14270.169342</td>
      <td>6.101210e+07</td>
      <td>4.420195</td>
      <td>4.508882</td>
      <td>0.210904</td>
      <td>3.358920</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2000.000000</td>
      <td>36.300000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>0.010000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>1.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>0.37000</td>
      <td>2.000000</td>
      <td>0.100000</td>
      <td>1.681350</td>
      <td>3.400000e+01</td>
      <td>0.100000</td>
      <td>0.100000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2004.000000</td>
      <td>63.100000</td>
      <td>74.000000</td>
      <td>0.000000</td>
      <td>0.877500</td>
      <td>4.685343</td>
      <td>77.000000</td>
      <td>0.000000</td>
      <td>19.300000</td>
      <td>0.000000</td>
      <td>78.000000</td>
      <td>4.26000</td>
      <td>78.000000</td>
      <td>0.100000</td>
      <td>463.935626</td>
      <td>1.957932e+05</td>
      <td>1.600000</td>
      <td>1.500000</td>
      <td>0.493000</td>
      <td>10.100000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2008.000000</td>
      <td>72.100000</td>
      <td>144.000000</td>
      <td>3.000000</td>
      <td>3.755000</td>
      <td>64.912906</td>
      <td>92.000000</td>
      <td>17.000000</td>
      <td>43.500000</td>
      <td>4.000000</td>
      <td>93.000000</td>
      <td>5.75500</td>
      <td>93.000000</td>
      <td>0.100000</td>
      <td>1766.947595</td>
      <td>1.386542e+06</td>
      <td>3.300000</td>
      <td>3.300000</td>
      <td>0.677000</td>
      <td>12.300000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2012.000000</td>
      <td>75.700000</td>
      <td>228.000000</td>
      <td>22.000000</td>
      <td>7.702500</td>
      <td>441.534144</td>
      <td>97.000000</td>
      <td>360.250000</td>
      <td>56.200000</td>
      <td>28.000000</td>
      <td>97.000000</td>
      <td>7.49250</td>
      <td>97.000000</td>
      <td>0.800000</td>
      <td>5910.806335</td>
      <td>7.420359e+06</td>
      <td>7.200000</td>
      <td>7.200000</td>
      <td>0.779000</td>
      <td>14.300000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2015.000000</td>
      <td>89.000000</td>
      <td>723.000000</td>
      <td>1800.000000</td>
      <td>17.870000</td>
      <td>19479.911610</td>
      <td>99.000000</td>
      <td>212183.000000</td>
      <td>87.300000</td>
      <td>2500.000000</td>
      <td>99.000000</td>
      <td>17.60000</td>
      <td>99.000000</td>
      <td>50.600000</td>
      <td>119172.741800</td>
      <td>1.293859e+09</td>
      <td>27.700000</td>
      <td>28.600000</td>
      <td>0.948000</td>
      <td>20.700000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.columns
```




    Index(['Country', 'Year', 'Status', 'Life expectancy ', 'Adult Mortality',
           'infant deaths', 'Alcohol', 'percentage expenditure', 'Hepatitis B',
           'Measles ', ' BMI ', 'under-five deaths ', 'Polio', 'Total expenditure',
           'Diphtheria ', ' HIV/AIDS', 'GDP', 'Population',
           ' thinness  1-19 years', ' thinness 5-9 years',
           'Income composition of resources', 'Schooling'],
          dtype='object')



### Initial Data Preparation

The original column names have extra spaces and other irregularities.  Let's clean those up, and also move the target variable to be the first column, for readability


```python
# rename so everything is snake_case
df = df.rename(columns={
    'Life expectancy ': 'Life_Expectancy',
    'Adult Mortality': 'Adult_Mortality',
    'infant deaths': 'Infant_Deaths',
    'percentage expenditure': 'Percentage_Expenditure',
    'Hepatitis B': 'Hepatitis_B',
    'Measles ': 'Measles',
    ' BMI ': 'BMI',
    'under-five deaths ': 'Under_five_Deaths',
    'Total expenditure': 'Total_Expenditure',
    'Diphtheria ': 'Diptheria',
    ' HIV/AIDS': 'HIV_AIDS',
    ' thinness  1-19 years': 'Thinness_1_19_years',
    ' thinness 5-9 years': 'Thinness_5_9_years',
    'Income composition of resources': 'Income_Composition_of_Resources'
})
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Country</th>
      <th>Year</th>
      <th>Status</th>
      <th>Life_Expectancy</th>
      <th>Adult_Mortality</th>
      <th>Infant_Deaths</th>
      <th>Alcohol</th>
      <th>Percentage_Expenditure</th>
      <th>Hepatitis_B</th>
      <th>Measles</th>
      <th>...</th>
      <th>Polio</th>
      <th>Total_Expenditure</th>
      <th>Diptheria</th>
      <th>HIV_AIDS</th>
      <th>GDP</th>
      <th>Population</th>
      <th>Thinness_1_19_years</th>
      <th>Thinness_5_9_years</th>
      <th>Income_Composition_of_Resources</th>
      <th>Schooling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Afghanistan</td>
      <td>2015</td>
      <td>Developing</td>
      <td>65.0</td>
      <td>263.0</td>
      <td>62</td>
      <td>0.01</td>
      <td>71.279624</td>
      <td>65.0</td>
      <td>1154</td>
      <td>...</td>
      <td>6.0</td>
      <td>8.16</td>
      <td>65.0</td>
      <td>0.1</td>
      <td>584.259210</td>
      <td>33736494.0</td>
      <td>17.2</td>
      <td>17.3</td>
      <td>0.479</td>
      <td>10.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Afghanistan</td>
      <td>2014</td>
      <td>Developing</td>
      <td>59.9</td>
      <td>271.0</td>
      <td>64</td>
      <td>0.01</td>
      <td>73.523582</td>
      <td>62.0</td>
      <td>492</td>
      <td>...</td>
      <td>58.0</td>
      <td>8.18</td>
      <td>62.0</td>
      <td>0.1</td>
      <td>612.696514</td>
      <td>327582.0</td>
      <td>17.5</td>
      <td>17.5</td>
      <td>0.476</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Afghanistan</td>
      <td>2013</td>
      <td>Developing</td>
      <td>59.9</td>
      <td>268.0</td>
      <td>66</td>
      <td>0.01</td>
      <td>73.219243</td>
      <td>64.0</td>
      <td>430</td>
      <td>...</td>
      <td>62.0</td>
      <td>8.13</td>
      <td>64.0</td>
      <td>0.1</td>
      <td>631.744976</td>
      <td>31731688.0</td>
      <td>17.7</td>
      <td>17.7</td>
      <td>0.470</td>
      <td>9.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Afghanistan</td>
      <td>2012</td>
      <td>Developing</td>
      <td>59.5</td>
      <td>272.0</td>
      <td>69</td>
      <td>0.01</td>
      <td>78.184215</td>
      <td>67.0</td>
      <td>2787</td>
      <td>...</td>
      <td>67.0</td>
      <td>8.52</td>
      <td>67.0</td>
      <td>0.1</td>
      <td>669.959000</td>
      <td>3696958.0</td>
      <td>17.9</td>
      <td>18.0</td>
      <td>0.463</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Afghanistan</td>
      <td>2011</td>
      <td>Developing</td>
      <td>59.2</td>
      <td>275.0</td>
      <td>71</td>
      <td>0.01</td>
      <td>7.097109</td>
      <td>68.0</td>
      <td>3013</td>
      <td>...</td>
      <td>68.0</td>
      <td>7.87</td>
      <td>68.0</td>
      <td>0.1</td>
      <td>63.537231</td>
      <td>2978599.0</td>
      <td>18.2</td>
      <td>18.2</td>
      <td>0.454</td>
      <td>9.5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>




```python
df.columns
```




    Index(['Country', 'Year', 'Status', 'Life_Expectancy', 'Adult_Mortality',
           'Infant_Deaths', 'Alcohol', 'Percentage_Expenditure', 'Hepatitis_B',
           'Measles', 'BMI', 'Under_five_Deaths', 'Polio', 'Total_Expenditure',
           'Diptheria', 'HIV_AIDS', 'GDP', 'Population', 'Thinness_1_19_years',
           'Thinness_5_9_years', 'Income_Composition_of_Resources', 'Schooling'],
          dtype='object')




```python
# reorder so life expectancy is the first column
cols = list(df.columns)
cols = [cols[3]] + cols[:3] + cols[4:]
df = df[cols]
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Life_Expectancy</th>
      <th>Country</th>
      <th>Year</th>
      <th>Status</th>
      <th>Adult_Mortality</th>
      <th>Infant_Deaths</th>
      <th>Alcohol</th>
      <th>Percentage_Expenditure</th>
      <th>Hepatitis_B</th>
      <th>Measles</th>
      <th>...</th>
      <th>Polio</th>
      <th>Total_Expenditure</th>
      <th>Diptheria</th>
      <th>HIV_AIDS</th>
      <th>GDP</th>
      <th>Population</th>
      <th>Thinness_1_19_years</th>
      <th>Thinness_5_9_years</th>
      <th>Income_Composition_of_Resources</th>
      <th>Schooling</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65.0</td>
      <td>Afghanistan</td>
      <td>2015</td>
      <td>Developing</td>
      <td>263.0</td>
      <td>62</td>
      <td>0.01</td>
      <td>71.279624</td>
      <td>65.0</td>
      <td>1154</td>
      <td>...</td>
      <td>6.0</td>
      <td>8.16</td>
      <td>65.0</td>
      <td>0.1</td>
      <td>584.259210</td>
      <td>33736494.0</td>
      <td>17.2</td>
      <td>17.3</td>
      <td>0.479</td>
      <td>10.1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>59.9</td>
      <td>Afghanistan</td>
      <td>2014</td>
      <td>Developing</td>
      <td>271.0</td>
      <td>64</td>
      <td>0.01</td>
      <td>73.523582</td>
      <td>62.0</td>
      <td>492</td>
      <td>...</td>
      <td>58.0</td>
      <td>8.18</td>
      <td>62.0</td>
      <td>0.1</td>
      <td>612.696514</td>
      <td>327582.0</td>
      <td>17.5</td>
      <td>17.5</td>
      <td>0.476</td>
      <td>10.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>59.9</td>
      <td>Afghanistan</td>
      <td>2013</td>
      <td>Developing</td>
      <td>268.0</td>
      <td>66</td>
      <td>0.01</td>
      <td>73.219243</td>
      <td>64.0</td>
      <td>430</td>
      <td>...</td>
      <td>62.0</td>
      <td>8.13</td>
      <td>64.0</td>
      <td>0.1</td>
      <td>631.744976</td>
      <td>31731688.0</td>
      <td>17.7</td>
      <td>17.7</td>
      <td>0.470</td>
      <td>9.9</td>
    </tr>
    <tr>
      <th>3</th>
      <td>59.5</td>
      <td>Afghanistan</td>
      <td>2012</td>
      <td>Developing</td>
      <td>272.0</td>
      <td>69</td>
      <td>0.01</td>
      <td>78.184215</td>
      <td>67.0</td>
      <td>2787</td>
      <td>...</td>
      <td>67.0</td>
      <td>8.52</td>
      <td>67.0</td>
      <td>0.1</td>
      <td>669.959000</td>
      <td>3696958.0</td>
      <td>17.9</td>
      <td>18.0</td>
      <td>0.463</td>
      <td>9.8</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59.2</td>
      <td>Afghanistan</td>
      <td>2011</td>
      <td>Developing</td>
      <td>275.0</td>
      <td>71</td>
      <td>0.01</td>
      <td>7.097109</td>
      <td>68.0</td>
      <td>3013</td>
      <td>...</td>
      <td>68.0</td>
      <td>7.87</td>
      <td>68.0</td>
      <td>0.1</td>
      <td>63.537231</td>
      <td>2978599.0</td>
      <td>18.2</td>
      <td>18.2</td>
      <td>0.454</td>
      <td>9.5</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 22 columns</p>
</div>



### Data Understanding

There are a lot of variables here!  Let's look at a correlation matrix to see which ones might be the most useful.  (Here we are looking for variables that are highly correlated with the target variable, but not highly correlated with other input variables)


```python
corr = df.corr()
mask = np.triu(np.ones_like(corr, dtype=np.bool))

fig1, ax1 = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=mask, ax=ax1)
```




    <matplotlib.axes._subplots.AxesSubplot at 0x12631bac8>




![png](Inferential_modeling_files/Inferential_modeling_14_1.png)


Ok, it looks like there are only a few that are highly positively correlated with life expectancy.  Let's make a pair plot of those.

(Note: we don't want to do a pair plot right from the outset because it would be too slow)


```python
positively_correlated_cols = ['Life_Expectancy','BMI', 'Income_Composition_of_Resources', 'Schooling']
positively_correlated_df = df[positively_correlated_cols]
sns.pairplot(positively_correlated_df);
```


![png](Inferential_modeling_files/Inferential_modeling_16_0.png)


### First Simple Model

Ok, it looks like the correlation with BMI is a little fuzzier than the others, so let's exclude it for now.  `Schooling` and `Income_Composition_of_Resources` are highly correlated with both life expectancy and each other, so let's only include one of them.  `Schooling` seems like a good choice because it would allow us to answer Question 5.


```python
fsm_df = df[["Schooling", "Life_Expectancy"]].copy()
fsm_df.dropna(inplace=True)
```


```python
fsm = ols(formula="Life_Expectancy ~ Schooling", data=fsm_df).fit()
```


```python
fsm.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>Life_Expectancy</td> <th>  R-squared:         </th> <td>   0.565</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.565</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3599.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 02 Mar 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>09:50:09</td>     <th>  Log-Likelihood:    </th> <td> -8964.3</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2768</td>      <th>  AIC:               </th> <td>1.793e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2766</td>      <th>  BIC:               </th> <td>1.794e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     1</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
      <td></td>         <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th> <td>   44.1089</td> <td>    0.437</td> <td>  100.992</td> <td> 0.000</td> <td>   43.252</td> <td>   44.965</td>
</tr>
<tr>
  <th>Schooling</th> <td>    2.1035</td> <td>    0.035</td> <td>   59.995</td> <td> 0.000</td> <td>    2.035</td> <td>    2.172</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>283.391</td> <th>  Durbin-Watson:     </th> <td>   0.267</td> 
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>1122.013</td> 
</tr>
<tr>
  <th>Skew:</th>          <td>-0.445</td>  <th>  Prob(JB):          </th> <td>2.28e-244</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 5.989</td>  <th>  Cond. No.          </th> <td>    46.7</td> 
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.



### Model Evaluation

Not too bad.  We are only explaining about 57% of the variance in life expectancy, but we only have one feature so far and it's statistically significant at an alpha of 0.05.

We could stop right now and say that according to our model:

 - A country with zero years of schooling on average is expected to have a life expectancy of 44.1 years
 - For each additional average year of schooling, we expect life expectancy to increase by 2.1 years

But before we move forward, let's also check for the assumptions of linear regression

#### Linearity

Linear regression assumes that the input variable linearly predicts the output variable.  We already qualitatively checked that with a scatter plot.  But I also think it's a good idea to use a statistical test.  This one is the [Rainbow test](https://www.tandfonline.com/doi/abs/10.1080/03610928208828423) which is available from the [diagnostic submodule of StatsModels](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.linear_rainbow.html#statsmodels.stats.diagnostic.linear_rainbow)


```python
rainbow_statistic, rainbow_p_value = linear_rainbow(fsm)
print("Rainbow statistic:", rainbow_statistic)
print("Rainbow p-value:", rainbow_p_value)
```

    Rainbow statistic: 1.291015978641167
    Rainbow p-value: 1.0575796565077053e-06


The null hypothesis is that the model is linearly predicted by the features, alternative hypothesis is that it is not.  Thus returning a low p-value means that the current model violates the linearity assumption.

#### Normality

Linear regression assumes that the residuals are normally distributed.  It is possible to check this qualitatively with a Q-Q plot, but this example shows how to assess it statistically.

The [Jarque-Bera](https://en.wikipedia.org/wiki/Jarque%E2%80%93Bera_test) test is performed automatically as part of the model summary output, labeled **Jarque-Bera (JB)** and **Prob(JB)**.

The null hypothesis is that the residuals are normally distributed, alternative hypothesis is that they are not.  Thus returning a low p-value means that the current model violates the normality assumption.

#### Homoscadasticity

Linear regression assumes that the variance of the dependent variable is homogeneous across different value of the independent variable(s).  We can visualize this by looking at the predicted life expectancy vs. the residuals.


```python
y = fsm_df["Life_Expectancy"]
y_hat = fsm.predict()
```


```python
fig2, ax2 = plt.subplots()
ax2.set(xlabel="Predicted Life Expectancy",
        ylabel="Residuals (Actual - Predicted Life Expectancy)")
ax2.scatter(x=y_hat, y=y-y_hat, color="blue", alpha=0.2)
```




    <matplotlib.collections.PathCollection at 0x1277dfd68>




![png](Inferential_modeling_files/Inferential_modeling_31_1.png)


Just visually inspecting this, it seems like our model over-predicts life expectancy between 60 and 70 years old in a way that doesn't happen for other age groups.  Plus we have some weird-looking data in the lower end that we might want to inspect.  Maybe there was something wrong with recording those values, or maybe there is something we can feature engineer once we have more independent variables.

Let's also run a statistical test.  The [Breusch-Pagan test](https://en.wikipedia.org/wiki/Breusch%E2%80%93Pagan_test) is available from the [diagnostic submodule of StatsModels](https://www.statsmodels.org/stable/generated/statsmodels.stats.diagnostic.het_breuschpagan.html#statsmodels.stats.diagnostic.het_breuschpagan)


```python
lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(y-y_hat, fsm_df[["Schooling"]])
print("Lagrange Multiplier p-value:", lm_p_value)
print("F-statistic p-value:", f_p_value)
```

    Lagrange Multiplier p-value: nan
    F-statistic p-value: 2.2825932549998897e-67


The null hypothesis is homoscedasticity, alternative hypothesis is heteroscedasticity.  Thus returning a low p-value means that the current model violates the homoscedasticity assumption

#### Independence

The independence assumption means that the independent variables must not be too collinear.  Right now we have only one independent variable, so we don't need to check this yet.

## Adding Features to the Model

So far, all we have is a simple linear regression.  Let's start adding features to make it a multiple regression.

First, I'll repeat the process of the highly positively correlated variables, but this time with the highly negatively correlated variables (based on looking at the correlation matrix)


```python
negatively_correlated_cols = [
    'Life_Expectancy',
    'Adult_Mortality',
    'HIV_AIDS',
    'Thinness_1_19_years',
    'Thinness_5_9_years'
]
negatively_correlated_df = df[negatively_correlated_cols]
sns.pairplot(negatively_correlated_df);
```


![png](Inferential_modeling_files/Inferential_modeling_38_0.png)


`Adult_Mortality` seems most like a linear relationship to me.  Also, the two thinness metrics seem to be providing very similar information, so we almost certainly should not include both

A quick experiment to try to flatten out that curve:


```python
fig3, ax3 = plt.subplots()

ax3.set(xlabel="Adult_Mortality", ylabel="Life_Expectancy")
ax3.scatter(x=np.sqrt(df["Adult_Mortality"]), y=df["Life_Expectancy"])
```




    <matplotlib.collections.PathCollection at 0x127ffe390>




![png](Inferential_modeling_files/Inferential_modeling_41_1.png)


This gives me straighter lines, but seems to indicate that we probably need at least two separate models to represent this data correctly.  However in the interest of time, I'm just going to assume the relationship is linear.


```python
model_2_df = df[["Life_Expectancy", "Schooling", "Adult_Mortality"]].copy()
model_2_df.dropna(inplace=True)
```


```python
model_2 = ols(formula="Life_Expectancy ~ Schooling + Adult_Mortality", data=model_2_df).fit()
```


```python
model_2.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>Life_Expectancy</td> <th>  R-squared:         </th> <td>   0.714</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.713</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   3443.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 02 Mar 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>10:05:02</td>     <th>  Log-Likelihood:    </th> <td> -8387.7</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2768</td>      <th>  AIC:               </th> <td>1.678e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2765</td>      <th>  BIC:               </th> <td>1.680e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     2</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>       <td>   56.0636</td> <td>    0.475</td> <td>  117.981</td> <td> 0.000</td> <td>   55.132</td> <td>   56.995</td>
</tr>
<tr>
  <th>Schooling</th>       <td>    1.5541</td> <td>    0.032</td> <td>   48.616</td> <td> 0.000</td> <td>    1.491</td> <td>    1.617</td>
</tr>
<tr>
  <th>Adult_Mortality</th> <td>   -0.0329</td> <td>    0.001</td> <td>  -37.803</td> <td> 0.000</td> <td>   -0.035</td> <td>   -0.031</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>537.142</td> <th>  Durbin-Watson:     </th> <td>   0.685</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>2901.045</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.813</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.745</td>  <th>  Cond. No.          </th> <td>1.02e+03</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.02e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



### Second Model Evaluation

Adding another feature improved the r-squared from 0.565 to 0.714

But let's also look at the model assumptions

#### Linearity


```python
rainbow_statistic, rainbow_p_value = linear_rainbow(model_2)
print("Rainbow statistic:", rainbow_statistic)
print("Rainbow p-value:", rainbow_p_value)
```

    Rainbow statistic: 1.0919639546889188
    Rainbow p-value: 0.05102555171520467


Assuming an alpha of 0.05, we are no longer violating the linearity assumption (just barely)

#### Normality

The **Jarque-Bera (JB)** output has gotten worse.  We are still violating the normality assumption.

#### Homoscadasticity


```python
y = model_2_df["Life_Expectancy"]
y_hat = model_2.predict()
```


```python
fig4, ax4 = plt.subplots()
ax4.set(xlabel="Predicted Life Expectancy",
        ylabel="Residuals (Actual - Predicted Life Expectancy)")
ax4.scatter(x=y_hat, y=y-y_hat, color="blue", alpha=0.2)
```




    <matplotlib.collections.PathCollection at 0x12840d6d8>




![png](Inferential_modeling_files/Inferential_modeling_52_1.png)



```python
lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(y-y_hat, model_2_df[["Schooling", "Adult_Mortality"]])
print("Lagrange Multiplier p-value:", lm_p_value)
print("F-statistic p-value:", f_p_value)
```

    Lagrange Multiplier p-value: 3.894654511131417e-47
    F-statistic p-value: 1.2521305604465352e-47


Both visually and numerically, we can see some improvement.  But we are still violating this assumption to a statistically significant degree.

#### Independence

You might have noticed in the regression output that there was a warning about the condition number being high.  The condition number is a measure of stability of the matrix used for computing the regression (we'll discuss this more in the next module), and a number above 30 can indicate strong multicollinearity.  Our output is way higher than that.

A different (more generous) measure of multicollinearity is the [variance inflation factor](https://en.wikipedia.org/wiki/Variance_inflation_factor).  It is available from the [outlier influence submodule of StatsModels](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html#statsmodels.stats.outliers_influence.variance_inflation_factor).


```python
rows = model_2_df[["Schooling", "Adult_Mortality"]].values

vif_df = pd.DataFrame()
vif_df["VIF"] = [variance_inflation_factor(rows, i) for i in range(2)]
vif_df["feature"] = ["Schooling", "Adult_Mortality"]

vif_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF</th>
      <th>feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1.937556</td>
      <td>Schooling</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1.937556</td>
      <td>Adult_Mortality</td>
    </tr>
  </tbody>
</table>
</div>



A "rule of thumb" for VIF is that 5 is too high, so I think it's reasonable to say that we are not violating the independence assumption, despite the high condition number.

## Adding a Categorical Value

This is less realistic than the previous steps but I wanted to give an example

In this dataset, we have a lot of numeric values (everything in that correlation matrix), but there are a few that aren't.  One example is `Status`


```python
model_3_df = df[["Life_Expectancy", "Schooling", "Adult_Mortality", "Status"]].copy()
model_3_df.dropna(inplace=True)
```


```python
model_3_df["Status"].value_counts()
```




    Developing    2304
    Developed      464
    Name: Status, dtype: int64




```python
sns.catplot(x="Status", y="Life_Expectancy", data=model_3_df, kind="box")
```




    <seaborn.axisgrid.FacetGrid at 0x128360d68>




![png](Inferential_modeling_files/Inferential_modeling_61_1.png)


It looks like there is a difference between the two groups that might be useful to include

There are only two categories, so we only need a `LabelEncoder` that will convert the labels into 1s and 0s.  If there were more than two categories, we would use a `OneHotEncoder`, which would create multiple columns out of a single column.


```python
label_encoder = LabelEncoder()
status_labels = label_encoder.fit_transform(model_3_df["Status"])
status_labels
```




    array([1, 1, 1, ..., 1, 1, 1])




```python
label_encoder.classes_
```




    array(['Developed', 'Developing'], dtype=object)



This is telling us that "Developed" is encoded as 0 and "Developing" is encoded as 1.  This means that "Developed" is assumed at the intercept.


```python
model_3_df["Status_Encoded"] = status_labels
model_3_df.drop("Status", axis=1)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Life_Expectancy</th>
      <th>Schooling</th>
      <th>Adult_Mortality</th>
      <th>Status_Encoded</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>65.0</td>
      <td>10.1</td>
      <td>263.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>59.9</td>
      <td>10.0</td>
      <td>271.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2</th>
      <td>59.9</td>
      <td>9.9</td>
      <td>268.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>3</th>
      <td>59.5</td>
      <td>9.8</td>
      <td>272.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>4</th>
      <td>59.2</td>
      <td>9.5</td>
      <td>275.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>2933</th>
      <td>44.3</td>
      <td>9.2</td>
      <td>723.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2934</th>
      <td>44.5</td>
      <td>9.5</td>
      <td>715.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2935</th>
      <td>44.8</td>
      <td>10.0</td>
      <td>73.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2936</th>
      <td>45.3</td>
      <td>9.8</td>
      <td>686.0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>2937</th>
      <td>46.0</td>
      <td>9.8</td>
      <td>665.0</td>
      <td>1</td>
    </tr>
  </tbody>
</table>
<p>2768 rows × 4 columns</p>
</div>




```python
model_3 = ols(formula="Life_Expectancy ~ Schooling + Adult_Mortality + Status_Encoded", data=model_3_df).fit()
```


```python
model_3.summary()
```




<table class="simpletable">
<caption>OLS Regression Results</caption>
<tr>
  <th>Dep. Variable:</th>     <td>Life_Expectancy</td> <th>  R-squared:         </th> <td>   0.718</td> 
</tr>
<tr>
  <th>Model:</th>                   <td>OLS</td>       <th>  Adj. R-squared:    </th> <td>   0.718</td> 
</tr>
<tr>
  <th>Method:</th>             <td>Least Squares</td>  <th>  F-statistic:       </th> <td>   2350.</td> 
</tr>
<tr>
  <th>Date:</th>             <td>Mon, 02 Mar 2020</td> <th>  Prob (F-statistic):</th>  <td>  0.00</td>  
</tr>
<tr>
  <th>Time:</th>                 <td>10:17:00</td>     <th>  Log-Likelihood:    </th> <td> -8364.0</td> 
</tr>
<tr>
  <th>No. Observations:</th>      <td>  2768</td>      <th>  AIC:               </th> <td>1.674e+04</td>
</tr>
<tr>
  <th>Df Residuals:</th>          <td>  2764</td>      <th>  BIC:               </th> <td>1.676e+04</td>
</tr>
<tr>
  <th>Df Model:</th>              <td>     3</td>      <th>                     </th>     <td> </td>    
</tr>
<tr>
  <th>Covariance Type:</th>      <td>nonrobust</td>    <th>                     </th>     <td> </td>    
</tr>
</table>
<table class="simpletable">
<tr>
         <td></td>            <th>coef</th>     <th>std err</th>      <th>t</th>      <th>P>|t|</th>  <th>[0.025</th>    <th>0.975]</th>  
</tr>
<tr>
  <th>Intercept</th>       <td>   58.9976</td> <td>    0.634</td> <td>   93.014</td> <td> 0.000</td> <td>   57.754</td> <td>   60.241</td>
</tr>
<tr>
  <th>Schooling</th>       <td>    1.4447</td> <td>    0.035</td> <td>   40.772</td> <td> 0.000</td> <td>    1.375</td> <td>    1.514</td>
</tr>
<tr>
  <th>Adult_Mortality</th> <td>   -0.0324</td> <td>    0.001</td> <td>  -37.395</td> <td> 0.000</td> <td>   -0.034</td> <td>   -0.031</td>
</tr>
<tr>
  <th>Status_Encoded</th>  <td>   -2.0474</td> <td>    0.296</td> <td>   -6.910</td> <td> 0.000</td> <td>   -2.628</td> <td>   -1.466</td>
</tr>
</table>
<table class="simpletable">
<tr>
  <th>Omnibus:</th>       <td>570.672</td> <th>  Durbin-Watson:     </th> <td>   0.678</td>
</tr>
<tr>
  <th>Prob(Omnibus):</th> <td> 0.000</td>  <th>  Jarque-Bera (JB):  </th> <td>2798.757</td>
</tr>
<tr>
  <th>Skew:</th>          <td>-0.899</td>  <th>  Prob(JB):          </th> <td>    0.00</td>
</tr>
<tr>
  <th>Kurtosis:</th>      <td> 7.586</td>  <th>  Cond. No.          </th> <td>1.45e+03</td>
</tr>
</table><br/><br/>Warnings:<br/>[1] Standard Errors assume that the covariance matrix of the errors is correctly specified.<br/>[2] The condition number is large, 1.45e+03. This might indicate that there are<br/>strong multicollinearity or other numerical problems.



### Third Model Evaluation

Adding another feature improved the r-squared a tiny bit from 0.714 to 0.718

Let's look at the model assumptions again

#### Linearity


```python
rainbow_statistic, rainbow_p_value = linear_rainbow(model_3)
print("Rainbow statistic:", rainbow_statistic)
print("Rainbow p-value:", rainbow_p_value)
```

    Rainbow statistic: 1.0769559317010546
    Rainbow p-value: 0.08416346182745871


Another small improvement

#### Normality

The **Jarque-Bera (JB)** output has gotten slightly better.  But we are still violating the normality assumption.

#### Homoscadasticity


```python
y = model_3_df["Life_Expectancy"]
y_hat = model_3.predict()
```


```python
fig5, ax5 = plt.subplots()
ax5.set(xlabel="Predicted Life Expectancy",
        ylabel="Residuals (Actual - Predicted Life Expectancy)")
ax5.scatter(x=y_hat, y=y-y_hat, color="blue", alpha=0.2)
```




    <matplotlib.collections.PathCollection at 0x12870b048>




![png](Inferential_modeling_files/Inferential_modeling_75_1.png)



```python
lm, lm_p_value, fvalue, f_p_value = het_breuschpagan(y-y_hat, model_3_df[["Schooling", "Adult_Mortality", "Status_Encoded"]])
print("Lagrange Multiplier p-value:", lm_p_value)
print("F-statistic p-value:", f_p_value)
```

    Lagrange Multiplier p-value: 6.828322778473366e-89
    F-statistic p-value: 9.274093378149953e-95


This metric got worse, although the plot looks fairly similar

#### Independence


```python
rows = model_3_df[["Schooling", "Adult_Mortality", "Status_Encoded"]].values

vif_df = pd.DataFrame()
vif_df["VIF"] = [variance_inflation_factor(rows, i) for i in range(3)]
vif_df["feature"] = ["Schooling", "Adult_Mortality", "Status_Encoded"]

vif_df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>VIF</th>
      <th>feature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>3.120074</td>
      <td>Schooling</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2.838221</td>
      <td>Adult_Mortality</td>
    </tr>
    <tr>
      <th>2</th>
      <td>4.518962</td>
      <td>Status_Encoded</td>
    </tr>
  </tbody>
</table>
</div>



The VIF metrics are getting higher, which means that there is stronger multicollinearity.  But we have still not exceeded the threshold of 5.

## Summary

We started with a baseline model where the only input feature was `Schooling`.  Our baseline model had an r-squared of 0.565.  This model violated the linearity (p < 0.001), normality (p < 0.001), and homoscadasticity (p < 0.001) assumptions of linear regression.  The independence assumption was met by default because there was only one input feature.

The final model for this lesson had three input features: `Schooling`, `Adult_Mortality`, and `Status_Encoded`.  It had an r-squared of 0.718.  This model did not violate the linearity assumption (p = 0.084), but it did violate the normality (p < 0.001) and homoscedasticity (p < 0.001) assumptions.  Based on the variance inflaction factor metric, it did not violate the independence assumption.

We are able to address the following questions from above:

*1. Does various predicting factors which has been chosen initially really affect the Life expectancy? What are the predicting variables actually affecting the life expectancy?*

With only 3 features we are able to explain about 71% of the variance in life expectancy.  This indicates that these factors truly are explanatory.  More analysis is required to understand how much additional explanatory power would be provided by incorporating additional features into the model.

*3. How does Infant and Adult mortality rates affect life expectancy?*

So far we have only investigated adult mortality.  The adult mortality rate ("probability of dying between 15 and 60 years per 1000 population") has a negative correlation with life expectancy.  For each increase of 1 in the adult mortality rate, life expectancy decreases by about .03 years.

*5. What is the impact of schooling on the lifespan of humans?*

In our latest model, we find that each additional year of average schooling is associated with 1.4 years of added life expectancy.  However it is challenging to interpret whether it is schooling that is actually having the impact.  Schooling is highly correlated with `Income_Composition_of_Resources` ("Human Development Index in terms of income composition of resources") so it is very possible that schooling is the result of some underlying factor that also impacts life expectancy, rather than schooling impacting life expectancy directly.

### Appendix

Things I have not done in this lesson, but that you should consider in your project:

 - More robust cleaning (possible imputation of missing values, principled exclusion of some data)
 - Feature scaling
 - Nearest-neighbors approach (requires more complex feature engineering)
 - Pulling information from external resources
 - Removing independent variables if you determine that they are causing too high of multicollinearity
 - Setting up functions so the code is not so repetitive
 
Also, I've included a dataset called `cars.csv` if you are interested in additional practice that does not use the King County Housing Data


```python
fig3, ax3 = plt.subplots()

ax3.set(xlabel="Status_Encoded", ylabel="Life_Expectancy")
ax3.scatter(x=model_3_df["Status_Encoded"], y=model_3_df["Life_Expectancy"])
```




    <matplotlib.collections.PathCollection at 0x1287d87b8>




![png](Inferential_modeling_files/Inferential_modeling_83_1.png)



```python

```
