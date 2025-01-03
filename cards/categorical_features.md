### Handling categorical features

---

### 1. **Label Encoding**

- **Description**: Converts each category into a unique integer.
- **Example**: For a feature "Color" with categories ["Red", "Blue", "Green"], label encoding might assign: Red=0, Blue=1, Green=2.
- **Use Case**: Suitable for ordinal categorical data where the categories have a natural order.
- Label encoding is intended to be used with converting target variables not input features, that's why it can not be used with column transformers, in that case only the ordinal encoder is to be used with the features


### 2. **One-Hot Encoding**

- **Description**: Converts each category into a new binary column, where only the column corresponding to the category is 1, and all others are 0.
- **Example**: For "Color" with categories ["Red", "Blue", "Green"], one-hot encoding creates three columns: [Color_Red, Color_Blue, Color_Green]. A "Red" entry would be [1, 0, 0].
- **Use Case**: Suitable for nominal categorical data where the categories have no inherent order.



### 3. **Ordinal Encoding**

- **Description**: Similar to label encoding but used when the categorical feature is ordinal, meaning it has a meaningful order.
- **Example**: For a feature "Size" with categories ["Small", "Medium", "Large"], ordinal encoding might assign: Small=1, Medium=2, Large=3.
- **Use Case**: Appropriate for ordinal features where the relative order matters.
  
