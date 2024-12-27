### Confusion Matrix, Precision & Recall

---

A confusion matrix is a table used to evaluate a classification model's performance. It summarizes the predicted and actual classifications of a model on a set of test data, showing the number of true positives, true negatives, false positives, and false negatives. It's a useful tool for understanding the strengths and weaknesses of a classifier, particularly in terms of how well it distinguishes between different classes. The total accuracy is not an good performance metric when there is an imbalance in the data set.

1. **Precision**: Precision is a metric that measures the proportion of **true positive predictions among all positive predictions** made by a classifier. Precision focuses on the accuracy of positive predictions.
2. **Recall (Sensitivity)**: Recall is a metric that measures the proportion of **true positive predictions among all actual positive instances** in the data. Recall focuses on how well the classifier identifies positive instances.
3. **F1 Score**: The F1 score is the harmonic mean of precision and recall, providing a single metric to evaluate a classifier's performance that balances both precision and recall. 

Precision and recall can be perfectly separable when the data is perfectly separable. These metrics are commonly used in evaluating binary classification models but can be extended to multi-class classification by averaging over all classes or using weighted averages.


![Confusion Matrix](../assets/images/confusionMatrxiUpdated.jpg)