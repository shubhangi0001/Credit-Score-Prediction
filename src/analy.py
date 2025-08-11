import pandas as pd

df = pd.read_csv("test_predictions.csv")

value_to_label = {
    -1.315771721676887: "Poor",
     0.1654527764396131: "Standard",
     1.6466772745561131: "Good"
}

df["Predicted_Label"] = df["Predicted_Credit_Score"].map(value_to_label)

df.to_csv("final_predictions.csv", index=False)
print("✅ Final predictions saved to 'final_predictions.csv'")

print("\nClass distribution:")
print(df["Predicted_Label"].value_counts())
