import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
#from hyperopt import fmin, tpe, hp
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

# 1. Reading data from CSV
def read_csv(file_path):
    return pd.read_csv(file_path)

# 2. Creating features
def create_features(data):
    # No feature creation for this example
    return data

# 3. Training a classifier model
def train_classifier(data):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    return model, accuracy

# 4. Hyperparameter tuning with Hyperopt
def objective(params):
    model = RandomForestClassifier(**params)
    score = cross_val_score(model, X, y, cv=5).mean()
    return -score  # Minimize negative accuracy

# 5. Evaluating the model on the test set
def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    return accuracy

file_path = "E:\MLOP\iris.csv"
data = read_csv(file_path)

print(data.head())

# if __name__ == "__main__":
# Load data
file_path = "E:\MLOP\iris.csv"
data = read_csv(file_path)

# Create features
data = create_features(data)

# Split data into features and target
X = data.drop('Species', axis=1)
y = data['Species']

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the pipeline
pipeline = Pipeline([
    ('preprocessor', ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), X.columns)
        ],
        remainder='passthrough'
    )),
    ('classifier', RandomForestClassifier())
])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
y_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Model accuracy on test set: {accuracy}")

#Brute force appraoch try with 2 for loops

# Hyperparameter tuning using Tree of Parzen Estimators (TPE)
#space = {
#    'n_estimators': hp.choice('n_estimators', range(10, 101)),
#    'max_depth': hp.choice('max_depth', range(1, 21))
#}

#best_params = fmin(fn=objective, space=space, algo=tpe.suggest, max_evals=100)

# Update the pipeline with the best hyperparameters


# Train the model with the best hyperparameters


# Evaluate the updated model
