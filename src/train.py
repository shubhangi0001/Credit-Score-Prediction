import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import joblib

base_dir = os.path.dirname(os.path.dirname(__file__)) 
train_path = os.path.join(base_dir, 'processed_data_train.csv')
test_path = os.path.join(base_dir, 'processed_data_test.csv')

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

train_df.columns = train_df.columns.str.strip()
test_df.columns = test_df.columns.str.strip()

target_column = 'Credit_Score'

X_train = train_df.drop(columns=[target_column])
y_train = train_df[target_column]

X_test = test_df.copy()  

cat_cols = X_train.select_dtypes(include=['object']).columns

label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X_train[col] = le.fit_transform(X_train[col].astype(str))
    X_test[col] = le.transform(X_test[col].astype(str))
    label_encoders[col] = le

target_encoder = LabelEncoder()
y_train = target_encoder.fit_transform(y_train)

clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

y_pred_train = clf.predict(X_train)
print("\nClassification Report (Train):")
target_names = [str(cls) for cls in target_encoder.classes_]
print(classification_report(y_train, y_pred_train, target_names=target_names))

model_dir = os.path.join(base_dir, 'models')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(clf, os.path.join(model_dir, 'credit_score_model.pkl'))
joblib.dump(label_encoders, os.path.join(model_dir, 'feature_encoders.pkl'))
joblib.dump(target_encoder, os.path.join(model_dir, 'target_encoder.pkl'))

print("\nModel and encoders saved to 'models/'")
