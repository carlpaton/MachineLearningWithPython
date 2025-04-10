# Foundation

- The end goal is to build a `model` / `engine` and train it on some input data. The more input data, the more accurate the `model` should be
- The `model` then learns pattens in the input data
- The `model` can then do predictions based on new data you give it, with a certain level of accuracy

## High Level Steps

Common workflows include

- **Import the data**, commonly CSV
- **Clean the data**
  - commonly used to remove _duplicates_, _irrelevant_ or _incomplete_ data
  - you dont always remove, you could just assign a default value
  - text based data needs to be represented by numerical values, example 0 for girl and 1 for boy, although in 2025 you could have -1 for cat 🐈
- **Split the data** into _training_ and _test_ sets, example 70% train and 30% test
  - training input data by convention is `X`
  - testing output data by convention is `y`
- **Create model** by seleting and algorithm to analyze the data
- **Train the model** by fitting the training data, it looks for patterns in the data
- **Make predictions** by passing test data and examining the output
- **Evaluate and improve** by measuring the accuracy of the output, if its bad, and most of the time when starting out it will be, the you need to
  - create a new model with a different algorithm
  - or fine tune model parameters to optimize the accuracy

## Choosing An Algorithm

There are many and certain `input data` / `problem you're trying to solve` work better with different algorithms, there are always pros and cons in terms of accuracy and performance.

![This image came from Infinite Codes](./algorithms.jpg)

- **Decision trees**:   
  - Used to classify data and predict numbers using a tree-like structure. They are easy to check and understand.
  - https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
- **Neural networks**: 
  - It works like the human brain with many connected nodes. They help to find patterns and are used in language processing, image and speech recognition, and creating images.
  - https://www.tensorflow.org/api_docs/python/tf/keras/applications/ResNet50
- **Linear regression**: It predicts numbers based on past data. For example, it helps estimate house prices in an area.
- **Logistic regression**: It predicts like "yes/no" answers and it is useful for spam detection and quality control.
- **Clustering**: It is used to group similar data without instructions and it helps to find patterns that humans might miss.
- **Random forests**: They combine multiple decision trees to improve predictions.

This list came from https://www.tutorialspoint.com/machine_learning/index.htm

Scikit-learn has a cheat sheet https://scikit-learn.org/stable/machine_learning_map.html

## Glossary Of Terms

- **[label](https://developers.google.com/machine-learning/glossary#label)**; In supervised machine learning, the "answer" or "result" portion of an example.
- **[feature](https://developers.google.com/machine-learning/glossary#feature)**; An input variable to a machine learning model.
- **over fitting**; training performance is good but validation performance is poor
- **under fitting**; both training performance and validation performance are poor

Google has an extensive glossary here https://developers.google.com/machine-learning/glossary