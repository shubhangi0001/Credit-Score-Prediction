import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import MultiLabelBinarizer
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Load data
X_train = pd.read_csv('../data/X_train.csv')
X_test = pd.read_csv('../data/X_test.csv')
y_train = pd.read_csv('../data/y_train.csv').squeeze()
y_test = pd.read_csv('../data/y_test.csv').squeeze()

# Encode 'Type_of_Loan' if it exists
if 'Type_of_Loan' in X_train.columns:
    print("🔄 Encoding 'Type_of_Loan'...")

    def preprocess_loan_column(df):
        return df['Type_of_Loan'].fillna('').str.replace(' and ', ',').str.split(',')

    mlb = MultiLabelBinarizer()

    X_train_loans = mlb.fit_transform(preprocess_loan_column(X_train))
    X_test_loans = mlb.transform(preprocess_loan_column(X_test))

    loan_columns = mlb.classes_
    X_train_loan_df = pd.DataFrame(X_train_loans, columns=loan_columns)
    X_test_loan_df = pd.DataFrame(X_test_loans, columns=loan_columns)

    # Drop original
    X_train = X_train.drop(columns=['Type_of_Loan'])
    X_test = X_test.drop(columns=['Type_of_Loan'])

    # Concatenate new encoded columns
    X_train = pd.concat([X_train.reset_index(drop=True), X_train_loan_df], axis=1)
    X_test = pd.concat([X_test.reset_index(drop=True), X_test_loan_df], axis=1)
else:
    print("❌ 'Type_of_Loan' not found. Proceeding without encoding.")

# Train model
clf = RandomForestRegressor(random_state=42)
clf.fit(X_train, y_train)

# Predict
y_pred = clf.predict(X_test)

# Evaluation
print("✅ Regression Evaluation:")
print("MSE:", mean_squared_error(y_test, y_pred))
print("R² score:", r2_score(y_test, y_pred))

# Save model
os.makedirs("../models", exist_ok=True)
joblib.dump(clf, '../models/rf_regressor.pkl')
print("✅ Model saved to '../models/rf_regressor.pkl'")

# Save predictions
pred_df = pd.DataFrame({
    'Actual': y_test,
    'Predicted': y_pred
})
pred_df.to_csv('../data/test_predictions_reg.csv', index=False)
print("📁 Predictions saved to '../data/test_predictions_reg.csv'")

# Plot Actual vs Predicted
plt.figure(figsize=(8, 6))
sns.scatterplot(x=y_test, y=y_pred, alpha=0.6)
plt.xlabel("Actual Credit Score")
plt.ylabel("Predicted Credit Score")
plt.title("Actual vs Predicted Credit Score")
plt.grid(True)
plt.tight_layout()
plt.savefig("../figures/actual_vs_predicted.png")
print("📊 Saved plot: '../figures/actual_vs_predicted.png'")

# Residual histogram
plt.figure(figsize=(8, 5))
sns.histplot(y_test - y_pred, bins=30, kde=True, color='salmon')
plt.title("Residual Error Distribution")
plt.xlabel("Residuals (Actual - Predicted)")
plt.tight_layout()
plt.savefig("../figures/residuals_hist.png")
print("📊 Saved plot: '../figures/residuals_hist.png'")
