# ML Model Comparison on Iris Dataset

Compared different linear and non-linear models using the popular Iris dataset.

## Major Dependencies

- Python3
- Pipenv

To install the dependencies, run the following command:

```bash
pipenv install
```

pipenv shell creates a new virtual env which makes sure that your local dependencies do not clash with this project.

```bash
pipenv shell
```

```bash
python <script-name>.py
```

To run the script.

## Results

(Might Vary depending on the random_state parameter and the stochastic nature)

- **Linear Models**:
  - Logistic Regression: Mean accuracy = 0.94
  - Linear Discriminant Analysis: Mean accuracy = 0.975

- **Non-Linear Models**:
  - K-Nearest Neighbors: Mean accuracy = 0.95
  - Decision Tree Classifier: Mean accuracy = 0.93
  - Gaussian Naive Bayes: Mean accuracy = 0.95
  - Support Vector Machine (Classifier): Mean accuracy = 0.983

## Predictions

Since SVM gave out the best results to match the dataset, After training the Support Vector Machine (SVM) model on the Iris dataset, I evaluated its performance on unseen data and analyzed the results using various metrics.

**Accuracy Score**: 0.9667

**Confusion Matrix**:

```
[[14  0  0]
 [ 0  7  1]
 [ 0  0  8]]
```

**Classification Report**:

```
                   precision    recall  f1-score   support
       Iris-setosa       1.00      1.00      1.00        14
   Iris-versicolor       1.00      0.88      0.93         8
    Iris-virginica       0.89      1.00      0.94         8
```

```
          accuracy                           0.97        30
         macro avg       0.96      0.96      0.96        30
      weighted avg       0.97      0.97      0.97        30
```

These results show that the SVM model predicts with high accuracy on the test data, with strong performance for the Iris-setosa class. However, it shows lower recall for Iris-versicolor, indicating that it may struggle slightly with identifying this class compared to the others. Overall, the model performed well with an accuracy of 0.97.
