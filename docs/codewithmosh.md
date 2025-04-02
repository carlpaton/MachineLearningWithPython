# Code With Mosh

Following [tutorial from Mosh](https://www.youtube.com/watch?v=7eh4d6sabA0) these are my notes with updates for 2025 version of the libraries

# Calculating The Accuracy

We should allocate 70, 80% of the data for training and use the remaining as testing

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre']) 
y = music_data['genre']

# X_train, X_test are input sets for training and testing
# y_train, y_test are output sets for training and testing, y_test are the expected values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2) # 20% of data to test

# create and train a model
model = DecisionTreeClassifier()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
score = accuracy_score(y_test, predictions)
score

# OUTPUTS: this will change because `train_test_split` selects random parts of the CSV and the sample size is only 18
# 1.0  -> means 100%
# 0.75 -> means  75%
# 0.5  -> means  50%
```

## Learning And Predicting

Pass entire dataset

```python
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

music_data = pd.read_csv('music.csv')
X = music_data.drop(columns=['genre']) 
y = music_data['genre']

# create and train a model
model = DecisionTreeClassifier()
model.fit(X.values, y)

# samples are passed to make predictions
# [21,1] is 21 year old male
# [22,0] is 22 year old female
predictions = model.predict([[21,1],[22,0]])
predictions

# OUTPUTS: this is expected based on the music.csv file
# array(['HipHop', 'Dance'], dtype=object)
```
