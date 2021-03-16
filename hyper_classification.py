"""
This is a simple Python script that demonstrates the impact that hyperparameter
tuning can have in a machine learning classifier.
"""

# Importing all required packages
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.datasets import load_digits

# Loading the digits dataset
digits = load_digits()

# Defining the steps and creating a pipeline
steps = [("scaler", StandardScaler()),
         ("SVM", SVC())]

pipeline = Pipeline(steps)

# Setting the hyperparameter space
parameters = {"SVM__C": [1, 10, 100],
              "SVM__gamma": [0.1, 0.01]}

# Defining our label and target variables
X = digits.data
y = digits.target

# Defining the testing and training variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21)

# Instantiating and fitting a GridSearchCV object
svm_cv = GridSearchCV(pipeline, parameters, cv=5)

svm_cv.fit(X_train, y_train)

# Fitting an unscaled logistic regression to the training set
svm_unscaled = SVC().fit(X_train, y_train)

# Printing the tuned parameters and score
print(f"Tuned SVM parameters: {svm_cv.best_params_}")
print(f"Tuned SVM score: {svm_cv.best_score_}")

# Printing the overall accuracy of the model
print(f"Scaled SVM accuracy: {svm_cv.score(X_test, y_test)}")
print(f"Unscaled SVM accuracy: {svm_unscaled.score(X_test, y_test)}")