# Image Recognition

Deep learning is a subset of machine learning that involves algorithms inspired by the structure and functions of a human brain, these are called artificial neural networks. Image recognition is a task commonly performed by using deep learning models.

This example uses ResNet but others exist like VGG and Inception.

See [tools and libraries](../docs/tools_and_libraries.md) to see what libraries are needed.

## Import Modules

Tensorflow is a comprehensive open source platform for machine learning, it provides the tools to build machine learning applications.

```python
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras import layers, models
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import splitfolders
import cv2
```

## Folders And Training Data

Download the sample [Agricultural crops image classification](https://www.kaggle.com/datasets/mdwaquarazam/agricultural-crops-image-classification) images and place them in the `input_folder`

```python
input_folder = '/Users/Gordon Freeman/ml-notebooks/Agricultural-crops'
output_folder = '/Users/Gordon Freeman/ml-notebooks/ImageRecognition'
```