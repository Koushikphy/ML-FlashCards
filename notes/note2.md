
## Bagging and Boosting


   



Both bagging and boosting are ensemble methods in machine learning that combine multiple models to improve overall performance. However, they work in fundamentally different ways.



### Bagging (Bootstrap Aggregating)

1. **How it Works**:  
   - **Data Resampling**: Multiple subsets of the dataset are created by sampling with replacement (bootstrap sampling).  
   - **Model Training**: A separate model (usually of the same type, like decision trees) is trained on each subset.  
   - **Aggregation**: The final prediction is made by combining the outputs of all models, typically through averaging (for regression) or majority voting (for classification).

2. **Key Features**:  
   - Models are trained **independently**.  
   - Reduces **variance** by averaging predictions.  
   - Best suited for high-variance, low-bias models (e.g., decision trees).

3. **Example Algorithm**:  
   Random Forest is a popular bagging-based algorithm where multiple decision trees are trained on bootstrapped datasets, and each tree considers a random subset of features for splits.

---

### Boosting


1. **How it Works**:  
   - **Sequential Learning**: Models are trained one after another, and each new model corrects the errors of its predecessor.  
   - **Weighted Updates**: More emphasis is placed on the samples that were misclassified or poorly predicted by earlier models.  
   - **Final Prediction**: A weighted combination of all models' outputs is used to make the final prediction.

2. **Key Features**:  
   - Models are trained **sequentially**.  
   - Reduces both **bias** and **variance**.  
   - Typically uses weak learners (e.g., shallow decision trees) and builds a strong learner by combining them.

3. **Example Algorithms**:  
   - **AdaBoost (Adaptive Boosting)**: Assigns higher weights to misclassified samples, forcing subsequent models to focus on these harder cases.  
   - **Gradient Boosting**: Optimizes the model by minimizing a loss function using gradient descent. Popular implementations include XGBoost, LightGBM, and CatBoost.




| **Aspect**         | **Bagging**                     | **Boosting**                  |
|---------------------|----------------------------------|--------------------------------|
| **Model Training**  | Independent                     | Sequential                    |
| **Goal**            | Reduce variance                | Reduce bias and variance      |
| **Focus**           | Equal focus on all samples      | Focus on difficult samples    |
| **Combination**     | Averaging or voting             | Weighted sum                  |
| **Complexity**      | Lower computational cost        | Higher computational cost     |
| **Overfitting**     | Less prone to overfitting       | May overfit if not regularized |
| **When to use**| Model has high variance | Model has high bias|



## Decision Trees

   


Decision Trees are supervised learning models used for both classification and regression tasks. They create a tree-like structure where each internal node represents a "decision" based on a feature, each branch represents an outcome of that decision, and each leaf node represents a class label or a numerical value. Key points include:

1. **Splitting Criteria:** Decision trees split nodes based on features that best separate the data into homogeneous subsets with respect to the target variable (e.g., Gini impurity for classification, variance reduction for regression). The gini impurity is calculated as $1-\sum (p_i)^2$ where $p_i$ is the probability of the sample being in category $i$. A particular node/split is chosen that minimizes the gini impurity. 
2. **Interpretability:** They are easy to understand and visualize, making them useful for explaining the decision-making process.
3. **Limitations:** Decision trees can overfit noisy data if not pruned properly and may not capture complex relationships as effectively compared to ensemble methods like Random Forests.

## Random Forest

   


Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees. Key points include:

1. **Bootstrap Sampling:** Random Forest builds each tree on a bootstrap sample (random sample with replacement) of the training data, ensuring diversity among the trees.
2. **Feature Randomness:** At each split in the tree, Random Forest considers only a subset of features (bagging), chosen randomly. This helps in reducing correlation among trees and improving generalization.



Reasons becaus random forest is preferred and often allow for stronger prediction than individual decision trees:

- Decision trees are prone to overfit whereas random forest generalizes better on unseen data as it uses randomness in feature selection and during sampling of the data. Therefore, random forests have lower variance compared to that of the decision tree without substantially increasing the error due to bias.
- Generally, ensemble models like random forests perform better as they are aggregations of various models (decision trees in the case of a random forest), using the concept of the “Wisdom of the crowd.”








## K-Nearest Neighbors 


   


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




## K-Means Clustering

   


K-Means is an unsupervised machine learning algorithm used for partitioning data into **K clusters**, where each cluster is represented by its centroid (mean point).


### How it works

1. **Initialize Centroids**: Select $K$ random points from the data as initial centroids.

2. **Assign Points to Clusters**:For each data point, calculate its distance to all centroids (commonly using Euclidean distance).Assign the point to the cluster of the closest centroid.

3. **Update Centroids**: Calculate the mean of all points in each cluster, and update the centroid to this mean value.

4. **Repeat**: Repeat the assignment and update steps until convergence i.e., when centroids no longer change significantly or a specified number of iterations is reached.


### Key points:

1. **Distance Metric**: Euclidean distance is commonly used to measure the distance between data points and centroids, but other metrics can be used depending on the application.

2. **Applications**: K-means clustering is widely used in various fields, including customer segmentation, image segmentation, document clustering, and anomaly detection.


3. **Evalutation metric** The Silhouette Coefficient is calculated using the mean intra-cluster distance (`a`) and the mean nearest-cluster distance (`b`) for each sample. The Silhouette Coefficient for a sample is `(b - a) / max(a, b)`.


4. **Choosing $K$: The Elbow Method**: Plot the total within-cluster sum of squares (inertia) against different values of $K$. Look for the "elbow" point where the rate of decrease slows down. This is often the optimal number of clusters.



### Advantages

- Simple and easy to implement.
- Scalable to large datasets.
- Works well when clusters are well-separated.


### Disadvantages

- Requires specifying $ K $ beforehand.
- Sensitive to:
  - Initial centroid placement (can lead to different results).
  - Outliers, which can skew clusters. Techniques like K-means++ are often used to improve initialization.
- Assumes clusters are spherical and evenly sized, which may not hold in real-world data.




## Support Vector Machine

   


SVM (Support Vector Machine) finds the optimal hyperplane in a high-dimensional space that best separates classes of data points. It aims to maximize the margin, which is the distance between the hyperplane and the nearest data points from each class, called support vectors.



### Key Concepts of SVM:
- **Hyperplane**: In SVM, the goal is to find the hyperplane (a decision boundary) that best separates the data points of different classes. For instance: In 2D space, the hyperplane is a line.In 3D space, it’s a plane. In higher dimensions, it’s a generalized hyperplane.
- **Support Vectors**: Support vectors are the data points that are closest to the hyperplane. These points are critical because they directly influence the position and orientation of the hyperplane.

- **Margin**: The margin is the distance between the hyperplane and the nearest data points from either class. SVM aims to maximize this margin, creating a decision boundary that generalizes well to unseen data.


### Kernel Trick
SVM can handle linearly separable and non-linearly separable datasets by using a kernel function. Common kernels include linear, polynomial, radial basis function (RBF), and sigmoid. The kernel function maps the input space into a higher-dimensional space, where it will be easier to find patterns in the data, making non-linear relationships separable by a hyperplane.

### Advantages
SVMs are effective in high-dimensional spaces and when the number of features exceeds the number of samples. They are also memory efficient due to their use of support vectors.

## Bayesian Inference

   


Bayesian inference is a statistical method based on **Bayes' Theorem**, which provides a way to update the probability estimate for a hypothesis as more evidence or data becomes available. It is a cornerstone of probabilistic reasoning and is widely used in machine learning, data science, and statistics.

### Bayes' Theorem Formula

$$P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)},$$


where:

- $P(H|E)$: **Posterior probability**. The probability of the hypothesis $H$ given the evidence $E$.
- $P(H)$: **Prior probability**. The initial probability of the hypothesis before observing any evidence.
- $P(E|H)$: **Likelihood**. The probability of observing the evidence $E$ given the hypothesis $H$ is true.
- $P(E)$: **Marginal likelihood** or evidence. The total probability of observing the evidence under all possible hypotheses.

#### **How It Works**
Bayesian inference adjusts the **prior belief** $P(H)$ based on new evidence $E$ to compute the **posterior belief** $P(H|E)$. It allows us to make predictions or decisions in the presence of uncertainty.



## Naive Bayes Classifier

   



The **Naive Bayes classifier** is a machine learning algorithm based on Bayes' Theorem, particularly suitable for classification tasks. It assumes that the features (input variables) are **conditionally independent** given the class label. This "naive" assumption makes the computation much simpler and more efficient, even though it may not hold perfectly in practice.

#### **Steps in Naive Bayes Classification**
1. **Compute Priors**:
   Calculate the prior probability for each class $P(C)$, where $C$ is the class label.

2. **Compute Likelihood**:
   Calculate the likelihood $P(X_i|C)$ for each feature $X_i$ in the dataset, given the class $C$.

3. **Apply Bayes' Theorem**:
   Use Bayes' Theorem to compute the posterior probability for each class:
   $$   P(C|X) \propto P(C) \prod_{i} P(X_i|C)$$

   Here, $P(C|X)$ is the posterior probability of class $C$ given the feature vector $X = (X_1, X_2, \dots, X_n)$.

4. **Predict Class**:
   Choose the class with the highest posterior probability:
   $$   \text{Predicted Class} = \arg\max_C P(C|X)$$




#### **Advantages**
- Simple and computationally efficient.
- Works well with high-dimensional data.
- Performs well with categorical data and text classification (e.g., spam filtering).

#### **Disadvantages**
- The assumption of conditional independence is often unrealistic.
- Performs poorly when features are highly correlated or when data is insufficient.



#### Common Use Cases
- **Text Classification**: Spam detection, sentiment analysis.
- **Medical Diagnosis**: Predicting diseases based on symptoms.
- **Recommender Systems**: Suggesting products based on user behavior.



#### Example

Suppose we want to classify an email as "Spam" or "Not Spam" based on the occurrence of certain words. Using Naive Bayes:  

1. Compute prior probabilities ($P(\text{Spam})$, $P(\text{Not Spam})$).  
2. Compute likelihoods ($P(\text{Word}| \text{Spam})$, $P(\text{Word}| \text{Not Spam})$).  
3. Calculate posterior probabilities for each class given the words in the email.  
4. Predict the class with the highest posterior probability.  


## Cross-validation

   


Cross-validation is a technique used to assess the performance of a learning model in several subsamples of training data.
It involves dividing the dataset into multiple subsets and systematically training and validating the model on different subsets to evaluate better estimate of the model's performance on unseen data. It reduces the risk of overfitting and ensure that the model generalizes well to new data.

### How it works
- **Split Data**: Divide the dataset into k subsets (folds).
- **Training and Validation**: Train the model on k-1 folds and validate it on the remaining fold.
- **Repeat**: Repeat this process k times, each time with a different fold as the validation set.
- **Aggregate Results**: Compute the average performance metric across all k iterations.


### Key points to note:
- Cross validation is done to estimate how well the pipeline generalizes to unseen data, which helps detect overfitting or data leakage before final training.
- It does not: (a) train a final model; (b) tune hyperparameters; (c) boost or combine results.
- When you pass a pipeline/model to a cross validation function, it internally creates separate, cloned copies of the pipeline and train and evaluate each clone independently. The original state of the pipeline/model at the end remain unchanged (not trained).

### Types of Cross-Validation
    
1. **K-Fold Cross-Validation**: The dataset is divided into k equally sized folds. The model is trained and validated k times, each time using a different fold as the validation set.
2. **Stratified K-Fold Cross-Validation**: Similar to k-fold, but ensures that each fold has the same proportion of classes as the original dataset. Useful for imbalanced datasets.
3. **Leave-One-Out Cross-Validation (LOOCV)**: Each instance in the dataset is used once as a validation set, and the remaining instances are used for training. Results in as many iterations as there are data points.
4. **Leave-P-Out Cross-Validation**: P data points are left out as the validation set, and the remaining points are used for training. Generalization of LOOCV where P is more than one.
5. **Time Series Cross-Validation**: Specifically designed for time series data where the order of data points matters. Ensures that training always occurs on past data and validation on future data to mimic real-world prediction. Example: Rolling-window cross-validation.

## Confusion Matrix, Precision & Recall

   


A confusion matrix is a table used to evaluate a classification model's performance. It summarizes the predicted and actual classifications of a model on a set of test data, showing the number of true positives, true negatives, false positives, and false negatives. It's a useful tool for understanding the strengths and weaknesses of a classifier, particularly in terms of how well it distinguishes between different classes. The total accuracy is not an good performance metric when there is an imbalance in the data set.

1. **Precision**: Precision is a metric that measures the proportion of **true positive predictions among all positive predictions** made by a classifier. Precision focuses on the accuracy of positive predictions.
2. **Recall (Sensitivity)**: Recall is a metric that measures the proportion of **true positive predictions among all actual positive instances** in the data. Recall focuses on how well the classifier identifies positive instances.
3. **F1 Score**: The F1 score is the harmonic mean of precision and recall, providing a single metric to evaluate a classifier's performance that balances both precision and recall. 

Precision and recall can be perfectly separable when the data is perfectly separable. These metrics are commonly used in evaluating binary classification models but can be extended to multi-class classification by averaging over all classes or using weighted averages.


#### ✅ Metric to Focus On:
- If **false positives are critical** (e.g., flagging a transaction as fraud, classifying a real email as spam), then **Precision** matters.
- If **false negatives are unacceptable** (e.g., missing of a deadly disease, missing a faulty product), then **Recall** or **F1 Score** is preferred.


![Confusion Matrix](./assets/images/confusionMatrxiUpdated.jpg)

## ROC Curve & AUC

   


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

## Bessel's Correction

   



Bessel's Correction is a technique used in statistics to correct the bias in the estimation of the population variance and standard deviation from a sample. It is particularly important when working with small sample sizes.  It involves using n-1 instead of n as the denominator in the formula for sample variance, where n is the sample size.

### Why Bessel's Correction is Needed:
When you calculate the variance or standard deviation of a sample, you are trying to estimate the variance or standard deviation of the entire population from which the sample was drawn. If you were to use the sample mean to calculate the variance, you would tend to **underestimate** the population variance, especially when the sample size is small. This happens because the sample mean is typically closer to the sample data points than the true population mean, which leads to smaller squared deviations.

### Formula for Variance with and without Bessel's Correction:
1. **Without Bessel's Correction** (biased estimator):
   The formula for the **sample variance** $s^2$ without Bessel's correction is:
   $$s^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$$
   where $x_i$ = individual data points, $\bar{x}$ = sample mean, $n$ = sample size.  
   This formula uses $n$ in the denominator, which tends to underestimate the population variance.

2. **With Bessel's Correction** (unbiased estimator):
   To correct this bias, Bessel’s correction uses $n - 1$ (degrees of freedom) instead of $n$. The corrected formula for the **sample variance** is:
   $$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$
   By dividing by $n - 1$, this correction makes the sample variance an **unbiased estimator** of the population variance. This ensures that, on average, the sample variance is equal to the true population variance when applied to multiple samples.

### Explanation:
- **Degrees of Freedom**: The term $n - 1$ represents the number of independent pieces of information available to estimate the population variance. The sample mean $\bar{x}$ is calculated from the data, so it is already constrained by the sample. This reduces the number of independent deviations by 1, hence $n - 1$ degrees of freedom.


## Hyperparameters

   


Hyperparameters are parameters that **are set prior to the training of a machine learning model**. Unlike model parameters, which are learned during training (e.g., weights in a neural network), hyperparameters are chosen by the data scientist or machine learning engineer based on prior knowledge, experience, or through experimentation. Here’s a concise explanation:

1. **Definition**:
    - Hyperparameters are configuration variables that determine the behavior and performance of a model.
    - They are not directly learned from the data but are set before training begins.
2. **Examples**:
    - **Learning Rate**: Controls how much to change the model in response to the estimated error each time the model weights are updated.
    - **Number of Trees (in Random Forest)**: Determines the number of decision trees to be used in the ensemble.
    - **Regularization Parameters**: Control the complexity of models, such as the penalty in ridge and lasso regression.
    - **Kernel Parameters (in SVM)**: Define the type of kernel function used and its specific parameters.
    - **Depth of Decision Trees**: Limits the maximum depth of decision trees in tree-based models like decision trees and random forests.
3. **Importance**:
    - Proper selection of hyperparameters can significantly impact the model's performance, convergence speed, and ability to generalize to new data.
    - Poor choices of hyperparameters can lead to overfitting or underfitting of the model.

## Hyperparameter Optimization

   


Hyperparameter optimization refers to the process of finding the best set of hyperparameters for a machine learning algorithm. It involves systematically searching through a predefined hyperparameter space and evaluating different combinations to identify the optimal configuration. Here’s a brief overview:

1. **Search Methods**:
    - **Grid Search**: Exhaustively searches through a manually specified subset of the hyperparameter space.
    - **Random Search**: Randomly samples hyperparameters from a predefined distribution.
    - **Bayesian Optimization**: Uses probabilistic models to predict the performance of hyperparameter combinations and focuses the search on promising regions.
2. **Evaluation**:
    - **Cross-Validation**: Typically used to evaluate each combination of hyperparameters to ensure robustness and avoid overfitting to the validation set.
3. **Tools and Libraries**:
    - **scikit-learn**: Provides tools for hyperparameter tuning, such as `GridSearchCV` and `RandomizedSearchCV`.
    - **Hyperopt**: Python library for optimizing over awkward search spaces with Bayesian optimization.
    - **Optuna**, **BayesianOptimization**: Other libraries that provide efficient hyperparameter optimization algorithms.
4. **Challenges**:
    - **Computational Cost**: Hyperparameter optimization can be computationally expensive, especially with large datasets and complex models.
    - **Curse of Dimensionality**: As the number of hyperparameters increases, the search space grows exponentially, making optimization more challenging.
5. **Best Practices**:
    - **Start Simple**: Begin with a broad search space and coarse resolution, then refine based on initial results.
    - **Domain Knowledge**: Use knowledge of the problem domain to narrow down the search space and prioritize hyperparameters likely to have the most impact.

## Central Limit Theorem

   


The Central Limit Theorem states that the sampling distribution of **the sample mean (or sum) approaches a normal distribution** as the sample size increases, regardless of the distribution of the population from which the samples are drawn. 

## Law of Large Numbers

   


The Law of Large Numbers states that as the number of trials or observations increases, the **sample mean will converge to the expected value (true mean) of the population**. In other words, with a larger sample size, the average of the observed results gets closer to the actual average of the entire population. 


< [Previous](note1.md) | [Next](note3.md) >