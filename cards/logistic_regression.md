### What is Logistic Regression

---

Logistic Regression is a statistical method used for binary classification problems, where the goal is to predict the probability of one of two possible outcomes (e.g., 0 or 1, true or false). The output is a value between 0 and 1, which can be interpreted as the probability of belonging to a certain class. Despite its name, it is a classification algorithm, not a regression one.

## **How It Works**
1. **Linear Model**: Logistic regression starts with a linear equation:
   $$z = w_1x_1 + w_2x_2 + \dots + w_nx_n + b$$
   where $x_1, x_2, \dots, x_n$ are the input features, $w_1, w_2, \dots, w_n$ are the weights, and $b$ is the bias.

2. **Sigmoid Function**: The linear output ($z$) is passed through a **sigmoid function** to map it into a probability range $[0, 1]$:
   $$P(y=1|x) = \frac{1}{1 + e^{-z}}$$
   The result is interpreted as the probability of the positive class ($y=1$).

3. **Decision Boundary**: A threshold (commonly 0.5) is applied to classify the output:
   - If $P(y=1|x) > 0.5$, classify as $y=1$.
   - Otherwise, classify as $y=0$.

## **Objective Function**
Logistic regression minimizes the **log-loss** (also called binary cross-entropy):
$$J(w, b) = - \frac{1}{m} \sum_{i=1}^{m} \left[ y_i \log(h(x_i)) + (1 - y_i) \log(1 - h(x_i)) \right]$$
where $h(x_i)$ is the predicted probability, $y_i$ is the actual label, and $m$ is the number of samples.

## **Key Features**
- **Linear Decision Boundary**: Logistic regression is a linear classifier, so it works well when the classes are linearly separable.
- **Probabilistic Output**: Provides probabilities, not just classifications, making it interpretable for applications like medical diagnostics.


