### Overfit & Underfit


---


Overfitting means that the model is doing well on the training data, but it does not generalize well on the test/validation data. It could be noticed when the training error is small, and the validation and test error is large. Overfitting happens when the model is too complex relative to the size of the data and its quality. This will result either in learning more about the pattern in the data noise or in very specific patterns in the data, which the model will not be able to generalize for new instances.

Here are possible solutions for overfitting:

- Simplify the model by decreasing the number of features or using regularization parameters.
- Collect more representative training data.
- Reduce the noise in the training data using data cleaning techniques.
-  Decrease the data mismatch using data preprocessing techniques.
- Use a validation set to detect when overfitting begins and stop the training.


Underfitting is, respectively, the opposite of overfitting. The model in this instance is too simple to learn any of the patterns in the training data. This could be seen when the training error is large, and the validation and test error is large.

Here are several possible solutions:

- Select a more complex model with more parameters.
- Reduce the regularization parameter if you are using it.
- Feed better features to the learning algorithm using feature engineering.