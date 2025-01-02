### Support Vector Machine

---

SVM (Support Vector Machine) finds the optimal hyperplane in a high-dimensional space that best separates classes of data points. It aims to maximize the margin, which is the distance between the hyperplane and the nearest data points from each class, called support vectors.



### Key Concepts of SVM:
- **Hyperplane**: In SVM, the goal is to find the hyperplane (a decision boundary) that best separates the data points of different classes. For instance: In 2D space, the hyperplane is a line.In 3D space, it’s a plane. In higher dimensions, it’s a generalized hyperplane.
- **Support Vectors**: Support vectors are the data points that are closest to the hyperplane. These points are critical because they directly influence the position and orientation of the hyperplane.

- **Margin**: The margin is the distance between the hyperplane and the nearest data points from either class. SVM aims to maximize this margin, creating a decision boundary that generalizes well to unseen data.


### Kernel Trick
SVM can handle linearly separable and non-linearly separable datasets by using a kernel function. Common kernels include linear, polynomial, radial basis function (RBF), and sigmoid. The kernel function maps the input space into a higher-dimensional space, where it will be easier to find patterns in the data, making non-linear relationships separable by a hyperplane.

### Advantages
SVMs are effective in high-dimensional spaces and when the number of features exceeds the number of samples. They are also memory efficient due to their use of support vectors.