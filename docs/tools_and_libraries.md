# Tools And Libraries

- Jupyter 
  - Integrated development environment (IDE) where you type your code https://jupyter.org/
  - It lets you execute partial lines of the code which is useful for ML
- Scikit-learn library
  - Provides common algorithms
- Numpy library
  - NumPy is an open source mathematical and scientific computing library for Python programming tasks. The name NumPy is shorthand for Numerical Python. The NumPy library offers a collection of high-level mathematical functions including support for multi-dimensional arrays, masked arrays and matrices. - [techtarget.com](https://www.techtarget.com/whatis/definition/What-is-NumPy-Explaining-how-it-works-in-Python)
- Pandas library
  - Data analysis which provides a concept called data frame, simliar to an Excel spreadsheet with rows and columns
- MatPlotLib library
  - Two dimentional library for creating graphs and plots
- Anaconda
  - Distrubtion software that downloads and installs Jupyter and the data science libraries mentioned above https://www.anaconda.com/download/success
- [Kaggle.com]()
  - Has data science projects and sample data sets
  - Example [Video game sales](https://www.kaggle.com/datasets/gregorut/videogamesales)

## Prerequisites

- Install Anaconda, I still needed to manually install Jupyter and required libraries
```
pip install notebook
pip install pandas
pip install sklearn
```
- Run notebook server, this should open http://localhost:8888/tree as the Jupyter dashboard
```
jupyter notebook
```
- Create Jupyter notebook, by default it will open to your users directory, I created the folder `ml-notebooks`, from the GUI selected `New` -> `Python 3 ipykernal`
- Rename the notebook from `Untitled` to be something related to your model, as always naming things is hard so I used `HelloWorld` like Mosh's example :D

### Jupyter Shortcuts

Command mode (press ESC to enable)

- `b`, insert new cell below
- `a`, insert a new cell above
- `d d`, delete cell
- `SHIFT TAB` on a method to show its tooltip
- `CTRL ENTER` with cell active, runs just that cell and doesnt add a new cell below