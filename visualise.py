import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load results
results = pd.read_csv("xgboost_results.csv")

sns.barplot(data=results[results["Metric"] == "Accuracy"],
            x="Dataset", y="Value", hue="Variation")
plt.title("Accuracy by Dataset and Variation")
plt.ylabel("Accuracy")
plt.show()

sns.boxplot(data=results[results["Metric"] == "Accuracy"],
            x="Dataset", y="Value", hue="Variation")
plt.title("Accuracy Distribution Across Variations")
plt.show()