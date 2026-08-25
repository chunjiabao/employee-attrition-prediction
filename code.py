import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (accuracy_score, f1_score, precision_score, classification_report, roc_auc_score)
import seaborn as sns
import matplotlib.pyplot as plt

# Upload data.csv into Google Colab
df = pd.read_excel('dataset.xlsx')

# Drop constant columns that make no meaning to analyse
df = df.drop(['EmployeeCount', 'StandardHours', 'Over18', 'EmployeeNumber', 'Year'], errors='ignore')


# Calculate Attrition Rate
num_yes = (df['Attrition'] == 'Yes').sum()
total_employees = len(df)
attrition_rate = round(num_yes / total_employees,4 )
print(df['Attrition'].value_counts())
print("Attrition rate:", attrition_rate)


# Encode the target variable 'Attrition'
df['Attrition'] = df['Attrition'].map({'Yes': 1, 'No': 0})


# Check for missing values
print(df.isnull().sum())


# Show Attrition value
sns.countplot(x='Attrition', data=df)
plt.title("Total Attrition Count 2023 & 2024")
plt.show()



X = df.drop('Attrition', axis=1)
y = df['Attrition']


# Onehot Encode column that have 'object' dtype
columns_to_encode = X.select_dtypes(include=['object']).columns
X = pd.get_dummies(X, columns=columns_to_encode)

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.3, random_state=0, stratify=y)

#Mean = 0 STD = 1
scaler = StandardScaler()
X_train_StandardScaler = scaler.fit_transform(X_train)
X_test_StandardScaler = scaler.transform(X_test)


feature_names = X.columns

# Convert back to DataFrame
X_train_scaled_df = pd.DataFrame(X_train_StandardScaler, columns=feature_names)
X_test_scaled_df = pd.DataFrame(X_test_StandardScaler, columns=feature_names)


mods = {
    "Logistic Regression": {
        "model": LogisticRegression(max_iter=5000, random_state=0, solver='liblinear'),
        "params": {'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2']},
        "X_train": X_train_scaled_df,
        "X_test": X_test_scaled_df
    },
    "K-Nearest Neighbors": {
        "model": KNeighborsClassifier(),
        "params": {'n_neighbors': [3, 5, 7, 9, 11], 'weights': ['uniform', 'distance']},
        "X_train": X_train_scaled_df,
        "X_test": X_test_scaled_df
    },
    "Gradient Boosting Classifier": {
        "model": GradientBoostingClassifier(random_state=0),
        "params": {'n_estimators': [100, 200], 'learning_rate': [0.1, 0.2]},
        "X_train": X_train_scaled_df,
        "X_test": X_test_scaled_df
    }
}


for name, config in mods.items():
    print("\n",name)

    # GridSearchCV for Hyperparameter Tuning
    grid_search = GridSearchCV(
        estimator=config["model"],
        param_grid=config["params"],
        cv=5,
        n_jobs=-1
    )

    # Fit the GridSearchCV
    grid_search.fit(config["X_train"], y_train)

    best_model = grid_search.best_estimator_ #Best model
    best_params = grid_search.best_params_ #Best param

    # Prediction of best model
    y_pred = best_model.predict(config["X_test"])

    # Calculate prob for ROC AUC
    y_pred_probability = best_model.predict_proba(config["X_test"])[:, 1]
    roc_auc = roc_auc_score(y_test, y_pred_probability)

    # Store results
    f1 = f1_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    accuracy = accuracy_score(y_test, y_pred)



    print(f"Best Parameters: {best_params}")
    print(f"F1-Score: {f1:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC Score: {roc_auc:.4f}")

importances = best_model.feature_importances_

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# Sort by importance descending
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)


print("Top 5 important features:")
print(feature_importance_df.head(5))