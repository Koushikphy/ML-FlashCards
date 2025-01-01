### K-Nearest Neighbors 


---

K-Nearest Neighbors (K-NN) is a **supervised learning algorithm** used for **classification** and **regression**. It is simple and intuitive, relying on the proximity of data points to make predictions. KNN works on the principle of similarity. It classifies or predicts a data point's target value based on how its nearest neighbors are classified or predicted. Its 

### K-NN Algorithm


- **For Classification**: Calculates the distance between the new data point and all points in the training set (e.g., usually Euclidean distance).Then, identify the \( K \)-nearest neighbors to the new point and assign the majority class among these neighbors to the new point.
- **For Regression**: Predicts the average (mean or median) value of the target variable among the \( K \)-nearest neighbors.


### Key Features

- **Lazy Learner**: No model is built during the training phase; the entire dataset is stored for prediction.
- **Distance Metric**: Common choices for measuring distance include: Euclidean Distance, Manhattan Distance, Cosine Similarity (for text or high-dimensional data).



### Advantages

- Simple and easy to implement.
- No assumptions about the data distribution.
- Works well with small datasets.


### Disadvantages

- **Computationally Expensive**: Requires calculating distances for all training points during prediction.
- **Sensitive to Noise**: Outliers can affect predictions, especially with small \( K \).
- **Curse of Dimensionality**: Performance degrades in high-dimensional spaces where distances become less meaningful.
- Requires careful scaling of features, as larger-scale features dominate distance calculations.


