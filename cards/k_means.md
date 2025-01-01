### K-Means Clustering

---

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


