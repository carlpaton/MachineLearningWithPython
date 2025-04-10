# Load Models

## music.joblib

This is a decision tree based model [fitted with this example](./../docs/codewithmosh.md).

```python
import joblib

model = joblib.load('music.joblib')

# [21,1] is 21 year old male
predictions = model.predict([[21,1]])
predictions
```

## Agricultural-crops.keras

This is a neural networks based model [fitted with this example](./../docs/image-recognition.md).

Load and setup the keras model.

```python
import tensorflow as tf
import cv2
import numpy as np
from tensorflow import keras

loaded_model = keras.models.load_model('Agricultural-crops.keras')

class_names = {
    0: 'Cherry',
    1: 'Coffee-plant',
    2: 'Cucumber',
    3: 'Fox_nut(Makhana)',
    4: 'Lemon',
    5: 'Olive-tree',
    6: 'Pearl_millet(bajra)',
    7: 'Tobacco-plant',
    8: 'almond',
    9: 'banana',
    10: 'cardamom',
    11: 'chilli',
    12: 'clove',
    13: 'coconut',
    14: 'cotton',
    15: 'gram',
    16: 'jowar',
    17: 'jute',
    18: 'maize',
    19: 'mustard-oil',
    20: 'papaya',
    21: 'pineapple',
    22: 'rice',
    23: 'soyabean',
    24: 'sugarcane',
    25: 'sunflower',
    26: 'tea',
    27: 'tomato',
    28: 'vigna-radiati(Mung)',
    29: 'wheat'
}

def predict_img(image, model):
    test_img=cv2.imread(image)                # read the image from the specified file path as an array
    test_img=cv2.resize(test_img, (224,224))  # resize to 224 by 224px to match the size the model was trained on
    test_img=np.expand_dims(test_img, axis=0) # numpy function to add extra dimensions to the image array
    result=model.predict(test_img)            # use trained model to make prediction
    r=np.argmax(result)                       # returns the index of the maxium value in the result array, 
                                              # this should correspond to the class with the highest probability
    print(class_names[r])
```

Call function

```python
predict_img(
    '/Users/Gordon Freeman/ml-notebooks/Agricultural-crops/Coffee-plant/images62.jpg',
    loaded_model
)
```