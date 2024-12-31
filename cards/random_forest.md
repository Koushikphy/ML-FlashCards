### Random Forest

---

Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees. Key points include:

1. **Bootstrap Sampling:** Random Forest builds each tree on a bootstrap sample (random sample with replacement) of the training data, ensuring diversity among the trees.
2. **Feature Randomness:** At each split in the tree, Random Forest considers only a subset of features (bagging), chosen randomly. This helps in reducing correlation among trees and improving generalization.



Reasons becaus random forest is preferred and often allow for stronger prediction than individual decision trees:

- Decision trees are prone to overfit whereas random forest generalizes better on unseen data as it uses randomness in feature selection and during sampling of the data. Therefore, random forests have lower variance compared to that of the decision tree without substantially increasing the error due to bias.
- Generally, ensemble models like random forests perform better as they are aggregations of various models (decision trees in the case of a random forest), using the concept of the “Wisdom of the crowd.”






