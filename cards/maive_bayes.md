### Naive Bayes Classifier

---


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
