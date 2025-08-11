import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

file_path = (r"D:/Credit_Score_Prediction/data/test_predictions_reg.csv")
df = pd.read_csv(file_path)

def convert_to_categories(values):
    """
    Convert continuous values to Standard, Good, Bad categories
    Adjust these thresholds based on your specific credit scoring system
    """
    categories = []
    for val in values:
        if val <= -0.5:  
            categories.append('Bad')
        elif val <= 0.5:  
            categories.append('Standard')
        else:
            categories.append('Good')
    return categories

def convert_to_categories_percentile(values):
    """
    Convert based on percentiles - you can adjust these percentiles
    """
    p33 = np.percentile(values, 33)
    p67 = np.percentile(values, 67)
    
    categories = []
    for val in values:
        if val <= p33:
            categories.append('Bad')
        elif val <= p67:
            categories.append('Standard')
        else:
            categories.append('Good')
    return categories

actual_categories = convert_to_categories(df['Actual'])
predicted_categories = convert_to_categories(df['Predicted'])

print("="*50)
print("CREDIT SCORE MODEL ACCURACY")
print("="*50)

accuracy = accuracy_score(actual_categories, predicted_categories)
print(f"Model Accuracy: {accuracy:.4f} ({accuracy:.2%})")

print(f"\nConfusion Matrix:")
cm = confusion_matrix(actual_categories, predicted_categories, labels=['Bad', 'Standard', 'Good'])
print("           Bad  Standard  Good")
for i, label in enumerate(['Bad', 'Standard', 'Good']):
    print(f"{label:>9}: {cm[i]}")

print(f"\nDetailed Report:")
print(classification_report(actual_categories, predicted_categories, labels=['Bad', 'Standard', 'Good']))

print(f"\nSample Conversions (first 10):")
print("Actual Value -> Category | Predicted Value -> Category")
for i in range(min(10, len(df))):
    print(f"{df['Actual'].iloc[i]:>12.3f} -> {actual_categories[i]:>8} | {df['Predicted'].iloc[i]:>12.3f} -> {predicted_categories[i]:>8}")

print(f"\nThreshold Info:")
print("Current thresholds: Bad (≤ -0.5), Standard (-0.5 to 0.5), Good (> 0.5)")
print("Adjust thresholds in convert_to_categories() function if needed")