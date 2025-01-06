### Hyperparameters

---

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