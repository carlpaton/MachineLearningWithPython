# Hello World

## Pandas

```python
import pandas as pd
df = pd.read_csv('vgsales.csv')
```

```python
# shows the shape `(16598, 11)` so thats 16k records and 11 columns
df.shape 

(16598, 11)
```

```python
# shows information about each column in the dataset when it has numerical values
# `count` tells us that `year` doesnt have values for all records so will need cleaning
# `mean` is the average value, in this dataset for `Rank` its meaningless but for `Year` it tells us the average year in the dataset 2006, this could be helpful based on the problem we are trying to solve
# `std` is the standard deviation, quantifies the amount of variation in the dataset
# `min` is the minimum value for that column, so here the oldest record is from 1980
# `25,50,75%` are the percentiles, so 2003 is in the 25th percentile
# `max` is the maximum value, so the newest record is from 2020 
df.describe()

	    Rank	        Year	        NA_Sales	    EU_Sales	    JP_Sales	    Other_Sales	    Global_Sales
count	16598.000000	16327.000000	16598.000000	16598.000000	16598.000000	16598.000000	16598.000000
mean	8300.605254	    2006.406443	    0.264667	    0.146652	    0.077782	    0.048063	    0.537441
std	    4791.853933	    5.828981	    0.816683	    0.505351	    0.309291	    0.188588	    1.555028
min	    1.000000	    1980.000000	    0.000000	    0.000000	    0.000000	    0.000000	    0.010000
25%	    4151.250000	    2003.000000	    0.000000	    0.000000	    0.000000	    0.000000	    0.060000
50%	    8300.500000	    2007.000000	    0.080000	    0.020000	    0.000000	    0.010000	    0.170000
75%	    12449.750000	2010.000000	    0.240000	    0.110000	    0.040000	    0.040000	    0.470000
max	    16600.000000	2020.000000	    41.490000	    29.020000	    10.220000	    10.570000	    82.740000
```

```python
# returns a 2 dimensional array showing samples of the top/bottom data
df.values

array([[1, 'Wii Sports', 'Wii', ..., 3.77, 8.46, 82.74],
       [2, 'Super Mario Bros.', 'NES', ..., 6.81, 0.77, 40.24],
       [3, 'Mario Kart Wii', 'Wii', ..., 3.79, 3.31, 35.82],
       ...,
       [16598, 'SCORE International Baja 1000: The Official Game', 'PS2',
        ..., 0.0, 0.0, 0.01],
       [16599, 'Know How 2', 'DS', ..., 0.0, 0.0, 0.01],
       [16600, 'Spirits & Spells', 'GBA', ..., 0.0, 0.0, 0.01]],
      shape=(16598, 11), dtype=object)
```