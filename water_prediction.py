import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (accuracy_score, classification_report,confusion_matrix)


import warnings
warnings.filterwarnings("ignore")


df = pd.read_csv("water_potability.csv")

# Separate X and y

X = df.drop('Potability', axis=1)
y = df['Potability']

numerical_features = X.columns
numerical_features

num_transformer = Pipeline(
    steps = [
        ('impute', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ]
)

preprocessor = ColumnTransformer(
    transformers= [
        ('num',num_transformer,numerical_features)
    ]
)


# Treain Test split
X_train,X_test, y_train, y_test = train_test_split(X,y, test_size = 0.2 , random_state=42)


rf_pipeline = Pipeline(
    steps = [
        ('preprocessing', preprocessor),
        ('model', RandomForestClassifier(n_estimators=100, random_state=42))
    ]
)

cv_scores = cross_val_score(rf_pipeline,
                            X_train,
                            y_train,
                            cv = 7,
                            scoring = 'accuracy')

print("Average of all fold scores", cv_scores.mean())
print("Standard deviation",cv_scores.std())


#  Hyperparameter Tuning
param_grid = {
    'model__n_estimators' : [100, 200],
    'model__max_depth' : [10, 20, 25],
    'model__min_samples_split' : [2,5]
}

grid_search = GridSearchCV(
    estimator = rf_pipeline,
    param_grid = param_grid,
    cv = 5,
    scoring = 'f1',
    n_jobs = -1,
    verbose = 2
)

grid_search.fit(X_train, y_train)

print(-grid_search.best_score_)

print(grid_search.best_params_)

#  Best Model Selection
best_model = grid_search.best_estimator_

# Model Performance Evaluation
y_test_pred = best_model.predict(X_test)


test_accu = accuracy_score(y_test, y_test_pred)

cla_report = classification_report(y_test, y_test_pred)

conf_matrix = confusion_matrix(y_test, y_test_pred)

print("Test Accuracy:", test_accu)
print("Classification Report:\n", cla_report)
print("Confusion Matrix", conf_matrix)


# =====================
# Save model (IMPORTANT)
# =====================

with open("water_rf_pipeline.pkl", "wb") as f:
    pickle.dump(best_model, f)

print("Random Forest pipeline saved as water_rf_pipeline.pkl")