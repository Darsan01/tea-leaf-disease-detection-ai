import os
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_DIR = "final_model_comparison"
os.makedirs(RESULTS_DIR, exist_ok=True)

# Final results from your 3 trained models
data = {
    "Model": ["Custom CNN", "MobileNetV2", "EfficientNetB0"],
    "Test Accuracy (%)": [96.54, 93.88, 91.76],
    "Overall Accuracy (%)": [96.81, 93.75, 91.76],
    "Macro Precision": [0.97, 0.94, 0.93],
    "Macro Recall": [0.97, 0.94, 0.92],
    "Macro F1-score": [0.97, 0.94, 0.92],
}

df = pd.DataFrame(data)

print("\nFinal Model Comparison")
print("======================\n")
print(df.to_string(index=False))

# Save table as CSV
df.to_csv(os.path.join(RESULTS_DIR, "model_comparison_table.csv"), index=False)

# Accuracy comparison graph
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["Test Accuracy (%)"])
plt.title("Model Test Accuracy Comparison")
plt.xlabel("Model")
plt.ylabel("Test Accuracy (%)")
plt.ylim(85, 100)

for i, value in enumerate(df["Test Accuracy (%)"]):
    plt.text(i, value + 0.3, f"{value:.2f}%", ha="center")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "model_accuracy_comparison.png"))
plt.close()

# F1-score comparison graph
plt.figure(figsize=(8, 5))
plt.bar(df["Model"], df["Macro F1-score"])
plt.title("Model Macro F1-score Comparison")
plt.xlabel("Model")
plt.ylabel("Macro F1-score")
plt.ylim(0.85, 1.00)

for i, value in enumerate(df["Macro F1-score"]):
    plt.text(i, value + 0.005, f"{value:.2f}", ha="center")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, "model_f1_score_comparison.png"))
plt.close()

# Save final conclusion
conclusion = """
Final Model Comparison Conclusion

Three models were trained and evaluated for tea leaf disease classification:
Custom CNN, MobileNetV2, and EfficientNetB0.

The Custom CNN achieved the best performance with a test accuracy of 96.54%
and a macro F1-score of 0.97. MobileNetV2 achieved 93.88% test accuracy,
while EfficientNetB0 achieved 91.76% test accuracy.

Based on the results, the Custom CNN was selected as the best model because
it achieved the highest accuracy and most balanced performance across all
four classes: Brown Blight, Healthy Leaf, Red Rust, and Red Spider Mite.
"""

with open(os.path.join(RESULTS_DIR, "final_comparison_conclusion.txt"), "w") as file:
    file.write(conclusion)

print("\nFiles saved in:", RESULTS_DIR)
print("1. model_comparison_table.csv")
print("2. model_accuracy_comparison.png")
print("3. model_f1_score_comparison.png")
print("4. final_comparison_conclusion.txt")