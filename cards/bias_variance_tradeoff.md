### Bias-Variance tradeoff

---

1. **Bias**: Bias refers to the error introduced by approximating a real-world problem with a simplified model. A high bias model is overly simplistic and tends to underfit the data, failing to capture important patterns and trends.
2. **Variance**: Variance measures the model's sensitivity to small fluctuations in the training data. A high variance model is complex and flexible, fitting the training data very closely but potentially overfitting noise and outliers.
3. **Tradeoff**: The bias-variance tradeoff implies that decreasing bias typically increases variance, and vice versa. The goal is to find a model that strikes a balance between bias and variance to achieve optimal predictive performance on new, unseen data.
4. **Implications**:
    - **Underfitting**: Models with high bias and low variance tend to underfit the training data, resulting in poor performance on both training and test datasets.
    - **Overfitting**: Models with low bias and high variance may overfit the training data, performing well on training data but poorly on test data due to capturing noise.
5. **Model Selection**: To find the optimal balance, techniques such as cross-validation, regularization, and ensemble methods (like bagging and boosting) are used:
    - **Regularization**: Introduces a penalty to the model complexity to reduce variance.
    - **Ensemble Methods**: Combine multiple models to reduce variance while maintaining low bias

The relationship between the bias of an estimator and its variance. Total prediction error = Bias$^2$+Variance+Iruducible error.
