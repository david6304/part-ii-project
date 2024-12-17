from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, confusion_matrix, f1_score, roc_auc_score
)
import xgboost as xgb
import bnlearn as bn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from dnn import DNN
import torch
from torch.utils.data import DataLoader, TensorDataset

# Load the 'alarm' dataset from bnlearn
data = bn.import_example('alarm')
print(data.head())
print(data.describe())

# Check for class imbalance in target variable 'dysp'
print("Class distribution in target variable:")
print(data['BP'].value_counts())

# Split data into features and target
X = data.drop(columns=['BP'])
y = data['BP']

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Logistic Regression Optimization ---
param_grid_lr = {
    'C': [0.01, 0.1, 1, 10],
    'solver': ['liblinear', 'saga']
}
grid_search_lr = GridSearchCV(LogisticRegression(max_iter=1000), param_grid_lr, cv=5, scoring='accuracy')
grid_search_lr.fit(X_train, y_train)
best_lr_model = grid_search_lr.best_estimator_

# Predict and evaluate Logistic Regression
y_pred_lr = best_lr_model.predict(X_test)
lr_probs = best_lr_model.predict_proba(X_test)[:, 1]

print("\nLogistic Regression Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_lr):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_lr, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_lr, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_lr, average='weighted'):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_lr)}")

# --- XGBoost Optimization ---
param_grid_xgb = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.1, 0.01, 0.001],
    'subsample': [0.7, 1.0],
    'n_estimators': [50, 100, 200]
}
xgb_model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss')

grid_search_xgb = GridSearchCV(xgb_model, param_grid_xgb, cv=5, scoring='accuracy')
grid_search_xgb.fit(X_train, y_train)
best_xgb_model = grid_search_xgb.best_estimator_

# Predict and evaluate XGBoost
y_pred_xgb = best_xgb_model.predict(X_test)
xgb_probs = best_xgb_model.predict_proba(X_test)[:, 1]

print("\nXGBoost Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_xgb):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_xgb, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_xgb, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_xgb, average='weighted'):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_xgb)}")

# --- Random Forest Optimization ---
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}
grid_search_rf = GridSearchCV(RandomForestClassifier(), param_grid_rf, cv=5, scoring='accuracy')
grid_search_rf.fit(X_train, y_train)
best_rf_model = grid_search_rf.best_estimator_

# Predict and evaluate Random Forest
y_pred_rf = best_rf_model.predict(X_test)
rf_probs = best_rf_model.predict_proba(X_test)[:, 1]

print("\nRandom Forest Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_rf, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_rf, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_rf, average='weighted'):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_rf)}")

# --- SVM Optimization ---
param_grid_svm = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf']
}
grid_search_svm = GridSearchCV(SVC(probability=True), param_grid_svm, cv=5, scoring='accuracy')
grid_search_svm.fit(X_train, y_train)
best_svm_model = grid_search_svm.best_estimator_

# Predict and evaluate SVM
y_pred_svm = best_svm_model.predict(X_test)
svm_probs = best_svm_model.predict_proba(X_test)[:, 1]

print("\nSVM Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_svm):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_svm, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_svm, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_svm, average='weighted'):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_svm)}")

# --- KNN Optimization ---
param_grid_knn = {
    'n_neighbors': [3, 5, 7],
    'weights': ['uniform', 'distance']
}
grid_search_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, cv=5, scoring='accuracy')
grid_search_knn.fit(X_train, y_train)
best_knn_model = grid_search_knn.best_estimator_

# Predict and evaluate KNN
y_pred_knn = best_knn_model.predict(X_test)
knn_probs = best_knn_model.predict_proba(X_test)[:, 1]

print("\nKNN Metrics:")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn):.4f}")
print(f"Precision: {precision_score(y_test, y_pred_knn, average='weighted'):.4f}")
print(f"Recall: {recall_score(y_test, y_pred_knn, average='weighted'):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_knn, average='weighted'):.4f}")
print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred_knn)}")

# Debugging: Compare probabilities
print("\nComparison of Predicted Probabilities (First 5 Instances):")
print(f"Logistic Regression: {lr_probs[:5]}")
print(f"XGBoost: {xgb_probs[:5]}")
print(f"Random Forest: {rf_probs[:5]}")
print(f"SVM: {svm_probs[:5]}")
print(f"KNN: {knn_probs[:5]}")

# --- Optional: Feature Importance ---
xgb_importance = best_xgb_model.feature_importances_
print("\nXGBoost Feature Importances:")
for feature, importance in zip(X.columns, xgb_importance):
    print(f"{feature}: {importance:.4f}")
