# Image Recognition

Deep learning is a subset of machine learning that involves algorithms inspired by the structure and functions of a human brain, these are called artificial neural networks. Image recognition is a task commonly performed by using deep learning models.

This example uses ResNet but others exist like VGG and Inception.

See the [prerequisites here](../docs/tools_and_libraries.md#prerequisites) to see what libraries I installed. Sometimes the `import/from` names are slighty different from the `install`.

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
import os
```

## Folders And Training Data

Download the sample [Agricultural crops image classification](https://www.kaggle.com/datasets/mdwaquarazam/agricultural-crops-image-classification) images and place them in the `input_folder`

```python
input_folder = '/Users/Gordon Freeman/ml-notebooks/Agricultural-crops'
output_folder = '/Users/Gordon Freeman/ml-notebooks/ImageRecognition' # save our file

split_ratio = (0.8, 0.1, 0.1) # 80% train, 10% validation, 10% test
splitfolders.ratio(
    input_folder, 
    output=output_folder, 
    seed=500, # random number generator, ensures the split is reproducible so running with the same seed means the same split
              # order types of ml models use `random-state`
    ratio=split_ratio, 
    group_prefix=None
)
```

## Define Parameters

```python
img_size = (224, 224) # resize the images to 224x224 pixels, this is a common size usef for deep learning
batch_size = 32       # models weight is updated after it processes 32 images

# data augmentation for the training data to expand the dataset with transformed versions, improves model generalization
train_datagen = ImageDataGenerator(
    preprocessing_function=preprocess_input, # resnet50 pre trained model
    rotation_range=20,                       # randomly rotate the image by up to 20 degrees
    width_shift_range=0.2,                   # randomly shift the image horizontally left/right by up to 20% of the width
    height_shift_range=0.2,                  # randomly shift the image vertically up/down by up to 20% of the height
    shear_range=0.2,                         # random shear transformations up to 20%
    zoom_range=0.2,                          # randomly zooms into the image up to 20%
    horizontal_flip=True,                    # randomly flips the image
    fill_mode='nearest'                      # when the image is rotated/shifted and a new pixel needs to be filled in, the nearest is used
)

test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) # data augmentation for test data (only rescaling)
valid_datagen = ImageDataGenerator(preprocessing_function=preprocess_input) # data augmentation for validation data (only rescaling)
```

## Create Generator

Create the generator that will read the images from traing,validation and test directorys, apply the specified augmentations and prepare them for the batches for the training model

```python
train_dir = os.path.join(output_folder, 'train')
test_dir = os.path.join(output_folder, 'test')
val_dir = os.path.join(output_folder, 'val')

train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical' # type of label array to be returned, categorical means the labels will be one hot encoded, useful for multiclass classification
)

test_data = test_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

valid_data = valid_datagen.flow_from_directory(
    test_dir,
    target_size=img_size,
    batch_size=batch_size,
    class_mode='categorical'
)

# OUTPUT
# Found 652 images belonging to 30 classes.
# Found 105 images belonging to 30 classes.
# Found 105 images belonging to 30 classes.
```

**Check random image**

This didnt work for me, tried `pip install pillow` but it was already installed.

```python
import sys
from PIL import Image

images, labels = next(valid_data) # get a batch of images / labels
idx = random.randit(0, images.shape[0] - 1) # Select a random image from the the batch

plt.imshow(images[idx])
plt.show()
```

## Load Model ResNet50

```python
from keras.applications.resnet import ResNet50 # Convolutional Neural Networks (CNN) that has been pre-trained on images
base_model = ResNet50(
    weights='imagenet',                       # use weights of the model that has been pre-trained
    include_top=False,                        # dont include the fully connected layers at the top of the network
                                              # the top refers to the classification layers that are normally at the end of our network
                                              # by excluding this we can add our own custom classification layers suitable for our problem

    input_shape=(img_size[0], img_size[1], 3) # shape of input images, they are expected 224 by 224px with 3 colour channels RGB (red/green/blue)
)

base_model.trainable=False # freeze convolutional base, meaning the weights of these layers will not be updated during training
                           # done to preserve the pre-trained weights and only train the newly added classification layers
                           # freezing the base model helps to leverage the features learnt from pre training without altering them

# OUTPUT
# Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
# 94765736/94765736 ━━━━━━━━━━━━━━━━━━━━ 5s 0us/step
```

Also see https://www.tensorflow.org/tutorials/images/transfer_learning

## Build Model

```python
model = models.Sequential([
    base_model,                            # pre trained ResNet50 model
    layers.GlobalAveragePooling2D(),       # used to replace fully connected layers in CNNs to reduce overfitting and the number of parameters
    layers.Dense(128, activation='relu'),  # fully connected dense layer with 128 neurons, `relu` (rectified linear unit) is the activation function
                                           # `relu` introduces nonlinearity enabling the model to learn more complex representations

    layers.Dropout(0.5),                   # randomly sets 50% of its input units to zero during each update
                                           # this also helps to prevent overfitting

    layers.Dense(30, activation='softmax') # another fully connected layer with 30 neurons
                                           # softmax activation function transforms raw output scores (logits) into a probability distribution
])
```

## Compile Model

```python
model.compile(
    optimizer='adam',                # updates the models weight during training to minimize the loss function
                                     # `adam` (adaotive moment estimation) is an advanced gradient descent algorithm that adjusts the learning rate for each parameter 
                                     # this done based on estimates for lower order moments
                                     # other optimizers exist but `adam` is widely used because its computationally efficient and requires less memory

    loss='categorical_crossentropy', # loss functions in ml measure how well the models prediction matches the true actual values
                                     # during training the optimizer tries to minimize the loss
                                     # other loss functions like `Mean Squared Error (MSE)` exist

    metrics=['accuracy']             # metrics evaluate the performance of the model
)
```

Also see https://keras.io/api/optimizers/adam/

## Fitting the model

- **over fitting**; training performance is good but validation performance is poor
- **under fitting**; both training performance and validation performance are poor

```python
model.fit(
    train_data,
    epochs=4,                  # epoch is one complete pass though the training data set, so 100 means do it 100 times (all batches)
                               # 100 is very common in examples (should take about 30 minutes on a standard desktop machine)
                               # the idea is with each epoch of time, the `accuracy` goes up and the `loss` goes down (see OUTPUT below)
    validation_data=valid_data
)

# OUTPUT
# Epoch 1/4
# 21/21 ━━━━━━━━━━━━━━━━━━━━ 37s 2s/step - accuracy: 0.1746 - loss: 2.9806 - val_accuracy: 0.3143 - val_loss: 2.3755
# Epoch 2/4
# 21/21 ━━━━━━━━━━━━━━━━━━━━ 38s 2s/step - accuracy: 0.2945 - loss: 2.5153 - val_accuracy: 0.4286 - val_loss: 2.0391
# Epoch 3/4
# 21/21 ━━━━━━━━━━━━━━━━━━━━ 40s 2s/step - accuracy: 0.3473 - loss: 2.2303 - val_accuracy: 0.5143 - val_loss: 1.7700
# Epoch 4/4
# 21/21 ━━━━━━━━━━━━━━━━━━━━ 40s 2s/step - accuracy: 0.4309 - loss: 1.9860 - val_accuracy: 0.5810 - val_loss: 1.6025
```

**Evaluate the model**

```python
test_loss, test_accuracy = model.evaluate(test_data)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# OUTPUT
# Test Accuracy: 22.86%   ~ with `epochs=1`
# Test Accuracy: 58.10%   ~ with `epochs=4` 
```

So in theory more epochs means the model gets smarter, based on the testing done by Karina the results were:

```
 25 epochs -> 80% accuracy
100 epochs -> 86% accuracy
```

## Classes Dictionary

There are 30 classes in the test dataset, so we need a dictionary of the classes, we need this to convert the index of the predicted class back to a human readable name.

```python
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
```

We then need a function to help us make the predictions

```python
def predict_img(image, model):
    test_img=cv2.imread(image)                # read the image from the specified file path as an array
    test_img=cv2.resize(test_img, (224,224))  # resize to 224 by 224px to match the size the model was trained on
    test_img=np.expand_dims(test_img, axis=0) # numpy function to add extra dimensions to the image array
    result=model.predict(test_img)            # use trained model to make prediction
    r=np.argmax(result)                       # returns the index of the maxium value in the result array, 
                                              # this should correspond to the class with the highest probability
    print(class_names[r])
```

**Test the function**

```python
predict_img(
    '/Users/Gordon Freeman/ml-notebooks/Agricultural-crops/Coffee-plant/images62.jpg',
    model
)

# OUTPUT
# 1/1 ━━━━━━━━━━━━━━━━━━━━ 2s 2s/step
# Cherry
```

## Persist the model

This is how `Keras/TensorFlow` models need to be saved because they it saves the model's architecture (the structure of the network), its weights (the learned parameters), and its optimizer state (how the model was trained). It preserves all aspects of the model, including custom layers and loss functions.

```python
model.save('Agricultural-crops.keras')
```

## Fine tune base model

The steps above have already frozen the convoltional base with of the ResNet50 model with `base_model.trainable=False`

Performance when using a substantial dataset can be tweaked as follows

- Use `linear rate scheduler` to help reduce the learning rate during training which could help with fine tuning
  - https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LinearLR.html
- Use `early stopping` which can prevent over fitting by stopping the training process with validation loss stops improving
- Use more `data augmentation`, some parameters were already applied above but we can also adjust things like `brightness` and or `contrast` of the images
- Use `batch normalization`
- If there is a lot of overfitting, try `L2 regularization`
  - L2 regularization, also known as ridge regression, is a technique used to prevent overfitting in machine learning models by adding a penalty term to the loss function that encourages small weights, thus preventing any single feature from dominating the prediction. 