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
 


