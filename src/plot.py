import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load CSV (make sure the path is correct)
df = pd.read_csv("final_predictions.csv")

# Plot
plt.figure(figsize=(8, 5))
ax = sns.countplot(x="Predicted_Credit_Score", data=df, palette="viridis")

# Annotate counts on bars
for p in ax.patches:
    count = int(p.get_height())
    ax.annotate(f"{count}", (p.get_x() + p.get_width() / 2., count), 
                ha='center', va='bottom')

plt.title("Distribution of Predicted Credit Score Classes")
plt.xlabel("Credit Score Category")
plt.ylabel("Number of Customers")
plt.tight_layout()
plt.show()
