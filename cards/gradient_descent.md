### Gradient Descent

---

Gradient descent is an optimization algorithm used to minimize a function by iteratively moving in the direction of steepest descent, as defined by the negative of the gradient.


### How it Works:
- **Initialize Parameters**: Start with an initial guess for the parameters (weights and biases, for instance). These are often set randomly.
- **Compute the Gradient**: The gradient is a vector of partial derivatives of the loss function with respect to each parameter. It points in the direction of the steepest increase in the loss.
- **Update Parameters**: Update the parameters by moving in the opposite direction of the gradient. This step can be mathematically expressed as:

    $$\theta= \theta-\alpha\cdot\nabla J(\theta)$$

    where:
    - $\theta$: The parameters.
    - $\alpha$: The learning rate, which controls the step size.
    - $\nabla J(\theta)$: The gradient of the loss function with respect to the parameters.



### Types of Gradient Descent:
- **Batch Gradient Descent**: Uses the entire dataset to compute the gradient at each step. It's computationally expensive for large datasets but provides a stable convergence.
- **Stochastic Gradient Descent (SGD)**: Uses a single data point to compute the gradient, leading to faster updates but more noise in convergence.
- **Mini-Batch Gradient Descent**: A middle ground where a small subset of data (mini-batch) is used to compute the gradient. It balances efficiency and stability.

### Challenges:
- Choosing the Learning Rate: If it's too large, you may overshoot the minimum; if it's too small, convergence may be slow.
- Local Minima and Saddle Points: The algorithm might get stuck in a local minimum or a saddle point instead of finding the global minimum.
- Vanishing/Exploding Gradients: In deep networks, gradients can become too small or too large, hindering learning.
  
### Extensions and Variants:
To address some challenges, variants like Momentum, RMSProp, and Adam add mechanisms to adapt the learning rate or smooth updates.
