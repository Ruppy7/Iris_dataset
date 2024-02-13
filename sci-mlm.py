from pandas import read_csv
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

#scatter_matrix(dataset)
#plt.show()

array = dataset.values
X = array[:, 0:4]
Y = array[:, 4]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20, random_state=1)

#Trying out 6 different algorithms (Logistic Regression, Linear Discriminant Analysis, K Nearest Neighbors, Classification and Regression Tress, Gaussian Naive Bayes, Support Vector Machines)

models = []
models.append(('LR', LogisticRegression(solver='liblinear', multi_class='ovr')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv)
    names.append(name)
    print('%s : %f (%f)' % (name, cv.mean(), cv.std()))
    
plt.boxplot(results, labels=names)
plt.title('Algorithm comparisons')
plt.show()