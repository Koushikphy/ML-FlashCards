### Fill Missing Data

---


1. **Remove Missing Values**: If the dataset is large and missing data is minimal, drop rows or columns with missing values.

2. **Imputation**:
    - **Mean/Median/Mode Imputation**: Replace missing values with the mean, median, or mode of the respective feature based on the distribution of the data.
    - **Forward/Backward Fill**: Fill missing values with the previous (forward fill) or next (backward fill) observed value, commonly used in time series data.
    - **K-Nearest Neighbors (KNN) Imputation**: Impute missing values based on the values of k-nearest neighbors in the feature space.
    - **Interpolation/Regression/ML Models**: Predict missing values using interpolation or ML models trained on the observed data.
3. **Using Domain Knowledge**: Leverage specific knowledge about the data or problem domain to fill in missing values in a more informed manner.

---

- **Mean Imputation:** When the data is approximately normally distributed, and the mean is a good representation of the central tendency. Can be heavily influenced by outliers, which may distort the imputed values if the data is skewed.
- **Median Imputation:** When The data is skewed or contains outliers, as the median is less affected by extreme values and better represents the central tendency in such cases. Robust to outliers and skewed distributions. May not be as effective in datasets with a large number of missing values if the median is not representative of the central tendency.
- **Mode Imputation:** When the feature is categorical or discrete, where the mode (most frequent value) is a meaningful representative value. Simple and preserves the most common category in the dataset. May not be appropriate for continuous data or features with many unique values, as it can lead to imbalanced distributions.