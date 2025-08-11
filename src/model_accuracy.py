import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score

# Load the test predictions dataset
file_path = r"D:\Credit_Score_Prediction\data\test_predictions_reg.csv"
df = pd.read_csv(file_path)

# Extract actual and predicted values
actual = df['Actual']
predicted = df['Predicted']

# Calculate accuracy using rounded values (for classification)
actual_rounded = np.round(actual).astype(int)
predicted_rounded = np.round(predicted).astype(int)
accuracy = accuracy_score(actual_rounded, predicted_rounded)

print(f"Credit Score Model Accuracy: {accuracy:.2%}")