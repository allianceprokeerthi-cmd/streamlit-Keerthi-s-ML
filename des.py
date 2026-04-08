# IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier

# LOAD DATA

df = pd.read_csv('/kaggle/input/datasets/ninzaami/loan-predication/train_u6lujuX_CVtuZ9i (1).csv')

# DATA PREPROCESSING

# Drop ID column
if 'Loan_ID' in df.columns:
    df.drop('Loan_ID', axis=1, inplace=True)

# Handle missing values
for col in df.columns:
    if df[col].dtype == 'object':
        df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        df[col].fillna(df[col].median(), inplace=True)

# Encode categorical variables
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])


# SPLIT DATA

X = df.drop('Loan_Status', axis=1)
y = df['Loan_Status']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42,)

 
#  MODELS
 
models = {
    "Logistic Regression": LogisticRegression(
        penalty='l2', C=0.5, class_weight='balanced',
        random_state=42, max_iter=200
    ),
    
    "Random Forest": RandomForestClassifier(
        n_estimators=200, max_depth=4, min_samples_split=10,
        class_weight='balanced', random_state=42
    ),
    
    "Gradient Boosting": GradientBoostingClassifier(
        n_estimators=100, learning_rate=0.05,
        max_depth=3, random_state=42
    ),
    
    "XGBoost": XGBClassifier(
        n_estimators=150, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric='logloss', random_state=42
    )
}

 
#  TRAIN, EVALUATE & PLOT
 
for name, model in models.items():
    print(f"\n{'='*50}")
    print(f"Model: {name}")
    print(f"{'='*50}")
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Classification Report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    print("Confusion Matrix:\n", cm)
    
    # Plot Confusion Matrix
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {name}')
    plt.show()

 
# CROSS VALIDATION (FINAL MODEL)
 
print("\n" + "="*50)
print("Final Model: Logistic Regression with CV")
print("="*50)

final_model = LogisticRegression(
    penalty='l2', C=0.5, class_weight='balanced',
    random_state=42, max_iter=200
)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

scores = cross_val_score(final_model, X, y, cv=cv, scoring='f1_macro')

print("F1 scores:", scores)
print("Average F1 score:", scores.mean())
