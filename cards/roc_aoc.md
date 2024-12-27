### ROC Curve & AUC

---

The ROC curve (Receiver Operating Characteristic curve) and AUC score (Area Under the ROC Curve) are tools used to evaluate the performance of binary classification models. Here’s a concise explanation:

1. **ROC Curve**:
    - **Definition**: The ROC curve is a graphical plot that illustrates **the diagnostic ability of a binary classifier as its discrimination threshold is varied**.
    - **X-axis**: False Positive Rate (FPR), which is  $\frac{\text{FP}}{\text{FP} + \text{TN}}$ , where FP is False Positives and TN is True Negatives.
    - **Y-axis**: True Positive Rate (TPR), which is $\frac{\text{TP}}{\text{TP} + \text{FN}}$ , where TP is True Positives and FN is False Negatives.
    - **Interpretation**: A diagonal line represents random guessing, and the ideal classifier would have a curve that goes straight up the Y-axis and then straight across the X-axis.
2. **AUC Score**:
    - **Definition**: The AUC score quantifies the overall performance of a binary classification model based on the ROC curve. It represents the area under the ROC curve.
    - **Interpretation**: A higher AUC score (closer to 1) indicates better discriminative ability of the model across all possible thresholds. An AUC score of 0.5 suggests the model performs no better than random guessing, and a score below 0.5 indicates worse than random guessing.

A similar plot can be done with a Precision-Recall (PR) plot, which is preferable when the dataset is highly skewed and better prediction of the minority class. Use the PR curve when you care more about the positive class (usually the minority class) and less about the negatives.