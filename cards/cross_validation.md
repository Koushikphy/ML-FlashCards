### Cross-validation

---

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