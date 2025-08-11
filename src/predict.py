import pandas as pd
import joblib
import os

# Paths
base_dir = os.path.dirname(os.path.dirname(__file__)) 

model_path = os.path.join(base_dir, 'models', 'credit_score_model.pkl')
encoder_path = os.path.join(base_dir, 'models', 'feature_encoders.pkl')
target_encoder_path = os.path.join(base_dir, 'models', 'target_encoder.pkl')

test_path = os.path.join(base_dir, 'processed_data_test.csv')
output_path = os.path.join(base_dir, 'test_predictions.csv')

print("Loading model and encoders...")
model = joblib.load(model_path)
feature_encoders = joblib.load(encoder_path)
target_encoder = joblib.load(target_encoder_path)

# Using test data 
print("Reading test data...")
test_df = pd.read_csv(test_path)
test_df.columns = test_df.columns.str.strip()

# Encode Features
print("Encoding categorical features...")
for col, le in feature_encoders.items():
    test_df[col] = le.transform(test_df[col].astype(str))

# Predictions
print(" Making predictions...")
pred_encoded = model.predict(test_df)
pred_labels = target_encoder.inverse_transform(pred_encoded)

print("Saving predictions to CSV...")
test_df['Predicted_Credit_Score'] = pred_labels
test_df.to_csv(output_path, index=False)

print(f"Predictions saved at: {output_path}")
