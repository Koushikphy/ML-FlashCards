### Decision Trees

---

Decision Trees are supervised learning models used for both classification and regression tasks. They create a tree-like structure where each internal node represents a "decision" based on a feature, each branch represents an outcome of that decision, and each leaf node represents a class label or a numerical value. Key points include:

1. **Splitting Criteria:** Decision trees split nodes based on features that best separate the data into homogeneous subsets with respect to the target variable (e.g., Gini impurity for classification, variance reduction for regression). The gini impurity is calculated as $1-\sum (p_i)^2$ where $p_i$ is the probability of the sample being in category $i$. A particular node/split is chosen that minimizes the gini impurity. 
2. **Interpretability:** They are easy to understand and visualize, making them useful for explaining the decision-making process.
3. **Limitations:** Decision trees can overfit noisy data if not pruned properly and may not capture complex relationships as effectively compared to ensemble methods like Random Forests.