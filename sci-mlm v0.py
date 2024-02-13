from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

#import pandas as pd

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

#pd.DataFrame(dataset).to_csv('output.csv', index=False)
""" Summarizing the DATASET
print(dataset.shape)           SHAPE
print(dataset.head(20))        HEAD/PEEK   
print(dataset.describe())      DESCRIPTIONS OF THE DATASET
print(dataset.groupby('class').size())  CLASS DISTRIBUTIONS
"""
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#dataset.hist()
scatter_matrix(dataset)
plt.show()