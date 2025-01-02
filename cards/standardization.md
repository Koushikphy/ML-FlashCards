### Data Standardization

---

Standardization and data scaling are essential preprocessing steps in many machine learning and statistical modeling tasks. These strategies ensure that features have comparable scales, which can improve the performance of many algorithms. Features with large ranges might overshadow others with smaller ranges. Standardization ensures that each feature contributes equally. Here are common strategies:

### 1. Standardization (Z-score Normalization)

- **Definition**: Transforms data to have a mean of zero and a standard deviation of one. $z = \frac{x - \mu}{\sigma}$
- **Use When**: The data follows a Gaussian (normal) distribution, and the goal is to ensure features are centered around zero with unit variance.
- **Applications**: Linear regression, logistic regression, SVM, k-means clustering.

### 2. Min-Max Scaling (Normalization)

- **Definition**: Scales data to a fixed range, usually [0, 1] or [-1, 1]. $x' = \frac{x - \min(x)}{\max(x) - \min(x)}$
- **Use When**: You want to scale features to a specific range.
- **Applications**: Neural networks, algorithms requiring bounded input values.

### 3. Robust Scaling

- **Definition**: Scales data based on percentiles, making it robust to outliers. $x' = \frac{x - \text{median}(x)}{\text{IQR}}$
**Use When**: The data contains significant outliers, and you want a method that reduces the influence of outliers.
- **Applications**: Any model sensitive to the scale of the data but robust to outliers.

### 4. MaxAbs Scaling

- **Definition**: Scales data to the range [-1, 1] by dividing each value by the maximum absolute value of the feature. $x' = \frac{x}{\max(|x|)}$
- **Use When**: The data contains features with varying magnitudes, and you want to preserve zero entries.
- **Applications**: Sparse data representations, such as TF-IDF matrices in text analysis.

### 5. Log Transformation

- **Definition**: Applies a logarithm to each data point to reduce skewness. $x' = \log(x + 1)$ (adding 1 to avoid log(0) issues).
- **Use When**: The data is highly skewed with long tails.
- **Applications**: Financial data, biological data.

### 6. Box-Cox Transformation

- **Definition**: Transforms data to approximate normality by applying a power transformation. $x' = \begin{cases}
\frac{x^\lambda - 1}{\lambda} & \text{if } \lambda \neq 0 \\
\log(x) & \text{if } \lambda = 0
\end{cases}$
- **Use When**: The data is not normally distributed, and normality is desired for modeling.
- **Applications**: Regression analysis, ANOVA.