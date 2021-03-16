"""
This is a simple Python script that demonstrates the impact that hyperparameter
tuning can have in a machine learning classifier.
"""

# Importing all required packages
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
import numpy as np
from gapminder import gapminder
import pandas as pd

# Loading the digits dataset and converting some columns to numerical values
gm = gapminder
gm = pd.get_dummies(gm, columns=["country", "continent"])

# Defining the steps and creating a pipeline
steps = [("imputer", SimpleImputer(missing_values=np.nan, strategy="mean")),
         ("scaler", StandardScaler()),
         ("elasticnet", ElasticNet())]

pipeline = Pipeline(steps)

# Setting the hyperparameter space
parameters = {"elasticnet__l1_ratio": np.linspace(0, 1, 30)}

# Defining our label and target variables
X = gm.drop("lifeExp", axis=1)
y = gm.lifeExp

# Defining the testing and training variables
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Instantiating and fitting a GridSearchCV object
elasticnet_cv = GridSearchCV(pipeline, parameters)

elasticnet_cv.fit(X_train, y_train)

# Fitting an unscaled logistic regression to the training set
elasticnet_unscaled = ElasticNet().fit(X_train, y_train)

# Calculating R squared
r2_scaled = elasticnet_cv.score(X_test, y_test)
r2_unscaled = elasticnet_unscaled.score(X_test, y_test)

# Printing the tuned parameters and score
print(f"Tuned elastic net alpha: {elasticnet_cv.best_params_}")

# Printing the overall accuracy of the model
print(f"Scaled elastic net R squared: {r2_scaled}")
print(f"Unscaled elastic net R squared: {r2_unscaled}")