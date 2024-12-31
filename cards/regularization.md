### Regularization

---

Regularization techniques in regression analysis are methods used to prevent overfitting and improve the generalization ability of models by adding a penalty to the loss function. The benefits of regularization techniques include improved model interpretability, reduced variance, and enhanced predictive performance on unseen data. 

 The two main types of regularization techniques are:

1. **Ridge Regression (L2 Regularization)**:
    - Ridge regression adds a penalty term proportional to the square of the coefficients (L2 norm) to the ordinary least squares (OLS) objective function ⇒ $\text{Loss} + \lambda \sum_{j=1}^{p} \beta_j^2$
    - Ridge regression is widely used in situations where multicollinearity is present or when the number of predictors (features) is large. It is a fundamental tool in regression analysis and machine learning for improving model robustness and interpretability.
2. **Lasso Regression (L1 Regularization)**:
    - Lasso regression adds a penalty term proportional to the absolute value of the coefficients (L1 norm) to the objective function ⇒ $\text{Loss} + \lambda \sum_{j=1}^{p} |\beta_j|$
    - It is particularly useful when dealing with high-dimensional datasets where many predictors may not contribute significantly to the model.
    - Unlike ridge regression, lasso can shrink coefficients all the way to zero, effectively performing feature selection by eliminating less important predictors.