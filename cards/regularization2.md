### Feature selection: Ridge vs Lasso

---

#### Ridge Regression: No Feature Selection

We start with the Ridge loss function for a single feature $x$ and coefficient $\beta$:

$$
L_2 = (y - x\beta)^2 + \lambda \beta^2
$$

Expanding the squared loss term:

$$
L_2 = y^2 - 2xy\beta + x^2\beta^2 + \lambda \beta^2
$$

Taking derivative w\.r.t. $\beta$ and setting to zero:

$$
\frac{dL_2}{d\beta} = -2xy + 2x^2\beta + 2\lambda\beta = 0
$$

Solving:

$$
2(x^2 + \lambda)\beta = 2xy \quad \Rightarrow \quad \beta = \frac{xy}{x^2 + \lambda}
$$

**Key Insight:**
The denominator $x^2 + \lambda$ is always **strictly positive**, and thus $\beta \neq 0$ unless $xy = 0$. Even for large $\lambda$, $\beta$ gets **shrunk**, but **never exactly zero** unless the correlation with $y$ vanishes.

---

#### Lasso Regression: Can Set Coefficients to Zero

Now consider the Lasso loss:

$$
L_1 = (y - x\beta)^2 + \lambda |\beta|
$$

Assume $\beta > 0$ for simplicity (similar logic applies for $\beta < 0$ or $\beta = 0$):

Expanding:

$$
L_1 = y^2 - 2xy\beta + x^2\beta^2 + \lambda \beta
$$

Taking derivative and setting to zero:

$$
\frac{dL_1}{d\beta} = -2xy + 2x^2\beta + \lambda = 0
$$

Solving:

$$
2x^2\beta = 2xy - \lambda \quad \Rightarrow \quad \beta = \frac{2xy - \lambda}{2x^2}
$$

**Key Insight:**
If $2xy \leq \lambda$, then $\beta \leq 0$. Since we assumed $\beta > 0$, this implies **no valid solution in the positive domain**, and the optimizer sets $\beta = 0$.

For small correlations $xy$, a moderate $\lambda$ is sufficient to push the entire numerator $(2xy - \lambda)$ to zero or negative, leading to **exactly zero** $\beta$. This is how **feature selection** occurs.

---

## ✅ Summary:

| Aspect              | Ridge                   | Lasso                               |
| ------------------- | ----------------------- | ----------------------------------- |
| Regularization      | $\lambda \|\beta\|_2^2$ | $\lambda \|\beta\|_1$               |
| Penalization effect | Shrinks coefficients    | Shrinks & can set to **exact zero** |
| Derivative behavior | Always smooth           | Has a **kink** at zero              |
| Geometry of penalty | Circular (smooth ball)  | Diamond-shaped (corners)            |
| Feature selection?  | ❌ No                    | ✅ Yes                               |

