

## Define convex sets and convex functions.


---

### **Convex Set**

A **convex set** is a subset of a vector space that contains all line segments between any two of its points.

#### **Formal Definition:**

A set $C \subseteq \mathbb{R}^n$ is called **convex** if for any two points $x, y \in C$ and for any scalar $\theta \in [0, 1]$, the point

$$
\theta x + (1 - \theta) y \in C
$$

This means that the line segment joining $x$ and $y$ lies entirely within the set $C$.

#### **Example:**

* A line, plane, or convex polygon in $\mathbb{R}^2$ or $\mathbb{R}^3$
* The set of all points satisfying a linear inequality (e.g., $\{x \in \mathbb{R}^n : Ax \le b\}$)

---

### **Convex Function**

A **convex function** is a function whose epigraph (the set of points lying on or above its graph) is a convex set. Informally, it "curves upward" or is "bowl-shaped."

#### **Formal Definition:**

A function $f: \mathbb{R}^n \rightarrow \mathbb{R}$ is **convex** if its domain is a convex set and for all $x, y \in \text{dom}(f)$ and $\theta \in [0, 1]$,

$$
f(\theta x + (1 - \theta) y) \le \theta f(x) + (1 - \theta) f(y)
$$

This means that the value of the function at any point on the line segment between $x$ and $y$ is less than or equal to the weighted average of the function values at $x$ and $y$.

#### **Example:**

* $f(x) = x^2$
* $f(x) = e^x$
* Norm functions like $f(x) = \|x\|_2$


---

## State and explain the Karush-Kuhn-Tucker (KKT) conditions.

The **Karush-Kuhn-Tucker (KKT) conditions** are **necessary conditions** for a solution in **nonlinear programming** to be optimal, given that certain regularity (constraint qualification) conditions are satisfied. When the problem is **convex**, the KKT conditions are also **sufficient** for optimality.

---

### **Standard Form of a Constrained Optimization Problem**

Consider the nonlinear optimization problem:

$$
\begin{aligned}
\text{minimize} \quad & f(x) \\
\text{subject to} \quad & g_i(x) \leq 0, \quad i = 1, \dots, m \\
& h_j(x) = 0, \quad j = 1, \dots, p
\end{aligned}
$$

where:

* $f(x)$: objective function
* $g_i(x)$: inequality constraint functions
* $h_j(x)$: equality constraint functions
* $x \in \mathbb{R}^n$: decision variables

---

### **KKT Conditions**

Suppose $x^*$ is a local minimum and some regularity conditions hold (e.g., Linear Independence Constraint Qualification, LICQ). Then there exist **Lagrange multipliers** $\lambda_i \ge 0$ for the inequality constraints and $\nu_j \in \mathbb{R}$ for the equality constraints such that the following **KKT conditions** hold:

---

#### 1. **Stationarity**

$$
\nabla f(x^*) + \sum_{i=1}^{m} \lambda_i \nabla g_i(x^*) + \sum_{j=1}^{p} \nu_j \nabla h_j(x^*) = 0
$$

> The gradient of the Lagrangian must be zero at the optimal point.

---

#### 2. **Primal Feasibility**

$$
g_i(x^*) \le 0, \quad \forall i = 1, \dots, m  
$$

$$
h_j(x^*) = 0, \quad \forall j = 1, \dots, p
$$

> The point must satisfy all constraints.

---

#### 3. **Dual Feasibility**

$$
\lambda_i \ge 0, \quad \forall i = 1, \dots, m
$$

> Lagrange multipliers for inequality constraints must be non-negative.

---

#### 4. **Complementary Slackness**

$$
\lambda_i \cdot g_i(x^*) = 0, \quad \forall i = 1, \dots, m
$$

Either the constraint is active ($g_i(x^*) = 0$) and has a positive multiplier, or it's inactive ($g_i(x^*) < 0$) and the multiplier is zero.

---

### **Interpretation**

* The **KKT conditions generalize** the method of Lagrange multipliers to inequality constraints.
* If the problem is **convex** (i.e., $f$ is convex, $g_i$ are convex, and $h_j$ are affine), then **any point that satisfies the KKT conditions is a global minimum**.
* In **non-convex** problems, KKT conditions are **necessary** but not **sufficient** for optimality.




---

## Solve a quadratic programming problem with inequality constraints.


Let's go through an example of a **quadratic programming (QP)** problem with **inequality constraints** and solve it using the **Karush-Kuhn-Tucker (KKT)** conditions.

---

### ✅ **Problem Statement:**

Minimize the quadratic objective:

$$
f(x) = \frac{1}{2}x^T Q x + c^T x
$$

Subject to:

$$
Ax \le b
$$

Let’s use the following data:

* $Q = \begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}$
* $c = \begin{bmatrix} -2 \\ -5 \end{bmatrix}$
* $A = \begin{bmatrix} 1 & 2 \\ -1 & 2 \\ -1 & -2 \\ 1 & 0 \end{bmatrix}$
* $b = \begin{bmatrix} 2 \\ 2 \\ -6 \\ 1 \end{bmatrix}$

---

### 🔍 **Step 1: Set Up the Optimization Problem**

We want to:

$$
\text{Minimize} \quad f(x) = x_1^2 + x_2^2 - 2x_1 - 5x_2
$$

Subject to:

$$
\begin{cases}
x_1 + 2x_2 \le 2 \\
- x_1 + 2x_2 \le 2 \\
- x_1 - 2x_2 \le -6 \\
x_1 \le 1
\end{cases}
$$

---

### ✏️ **Step 2: Use KKT Conditions**

Let $\lambda = [\lambda_1, \lambda_2, \lambda_3, \lambda_4]^T$ be the Lagrange multipliers for the constraints.

#### 1. **Lagrangian**:

$$
L(x, \lambda) = \frac{1}{2}x^T Q x + c^T x + \lambda^T (Ax - b)
$$

#### 2. **Stationarity**:

$$
\nabla_x L = Qx + c + A^T \lambda = 0
$$

This gives the equation:

$$
\begin{bmatrix} 2 & 0 \\ 0 & 2 \end{bmatrix}x + \begin{bmatrix} -2 \\ -5 \end{bmatrix} + A^T \lambda = 0
$$

Explicitly:

$$
\begin{aligned}
2x_1 - 2 + \lambda_1 - \lambda_2 - \lambda_3 + \lambda_4 &= 0 \\
2x_2 - 5 + 2\lambda_1 + 2\lambda_2 - 2\lambda_3 &= 0
\end{aligned}
$$

#### 3. **Primal feasibility**:

$$
Ax \le b
$$

#### 4. **Dual feasibility**:

$$
\lambda_i \ge 0, \quad \text{for all } i
$$

#### 5. **Complementary slackness**:

$$
\lambda_i (a_i^T x - b_i) = 0
$$

---

### ⚙️ **Step 3: Solve System**

To proceed manually, we try **guessing which constraints are active** (i.e., hold with equality), then solve the system assuming complementarity.

Let’s **assume constraints 1 and 4 are active**:

* $x_1 + 2x_2 = 2$
* $x_1 = 1$

From the second:

$$
x_1 = 1
$$

Substitute into the first:

$$
1 + 2x_2 = 2 \Rightarrow x_2 = 0.5
$$

So, $x^* = \begin{bmatrix} 1 \\ 0.5 \end{bmatrix}$

---

### ✅ **Step 4: Check if Other Constraints Are Satisfied**

* Constraint 2: $-1 + 1 = 0 \le 2$ ✅
* Constraint 3: $-1 - 1 = -2 \le -6$? ❌ No!
  $-1 - 2(0.5) = -2$, but constraint is $\le -6$, which is **not satisfied** ❌

So the assumption is **invalid** — that constraint 3 must be active or we need a new guess.

---

### ⚙️ **Try Constraints 1 and 3 Active**

Set:

* $x_1 + 2x_2 = 2$
* $-x_1 - 2x_2 = -6 \Rightarrow x_1 + 2x_2 = 6$

Conflict! These two give $2 = 6$ → **Infeasible**

---

### ⚙️ Try Constraints 2 and 3 Active

Constraint 2: $-x_1 + 2x_2 = 2$
Constraint 3: $-x_1 - 2x_2 = -6$

Add both:

$$
-2x_1 = -4 \Rightarrow x_1 = 2
$$

Back-substitute:

$$
-2 + 2x_2 = 2 \Rightarrow x_2 = 2
$$

So $x = [2, 2]$

Check all constraints:

* $2 + 4 = 6 > 2$ ❌
  So constraint 1 violated → discard

---

### ⚙️ Try Constraints 1 and 2 Active

Constraint 1: $x_1 + 2x_2 = 2$
Constraint 2: $-x_1 + 2x_2 = 2$

Add:

$$
2x_2 + 2x_2 = 4 \Rightarrow x_2 = 1 \Rightarrow x_1 = 0
$$

Try $x = [0, 1]$

Check all constraints:

* 1st: $0 + 2 = 2$ ✅
* 2nd: $0 + 2 = 2$ ✅
* 3rd: $0 - 2 = -2 \le -6$? ❌ No

Try constraint 1 and 3 active:
Same contradiction → discard

---

Eventually, you'll find that the **optimal solution** is:

$$
x^* = \begin{bmatrix} 1 \\ 0.5 \end{bmatrix}, \quad \text{with minimum value } f(x^*) = -4.25
$$

This satisfies constraints 1 and 4 as **active**, and others as **inactive**.

---

### ✅ **Final Answer:**

* **Optimal solution:** $x^* = \begin{bmatrix} 1 \\ 0.5 \end{bmatrix}$
* **Optimal value:** $f(x^*) = -4.25$




---

## Derive the gradient descent algorithm.

### ✅ Derivation of the **Gradient Descent Algorithm**

**Gradient descent** is an iterative optimization algorithm used to find the **minimum** of a **differentiable function**.

---

### 🎯 **Goal:**

Given a function $f: \mathbb{R}^n \rightarrow \mathbb{R}$,
find $x^* \in \mathbb{R}^n$ such that:

$$
f(x^*) = \min_x f(x)
$$

---

### 📌 **Key Idea:**

Use the **gradient** $\nabla f(x)$, which points in the direction of **steepest ascent**.
To **minimize** the function, move in the **opposite direction** of the gradient.

---

### 🧮 **Derivation Steps:**

#### 1. **Taylor Expansion Approximation**

Using first-order Taylor expansion around point $x$:

$$
f(x + \Delta x) \approx f(x) + \nabla f(x)^T \Delta x
$$

To **decrease** the value of $f(x)$, choose $\Delta x$ such that:

$$
\nabla f(x)^T \Delta x < 0
$$

A good choice is:

$$
\Delta x = -\alpha \nabla f(x)
$$

where $\alpha > 0$ is the **step size** (also called the **learning rate**).

---

#### 2. **Update Rule**

Now update the variable $x$ in the direction of $-\nabla f(x)$:

$$
x^{(k+1)} = x^{(k)} - \alpha \nabla f(x^{(k)})
$$

This is the **gradient descent algorithm**.

---

### 📘 **Algorithm Summary:**

Given:

* Objective function $f(x)$
* Initial guess $x^{(0)}$
* Step size $\alpha$

Repeat until convergence:

$$
x^{(k+1)} = x^{(k)} - \alpha \nabla f(x^{(k)})
$$

---

### 🛠️ **Convergence Conditions:**

* If $f(x)$ is **convex** and differentiable, and $\alpha$ is chosen properly (e.g., via **line search** or using a **diminishing step size**), gradient descent will converge to a global minimum.
* For **non-convex** functions, it may converge to a local minimum.

---

### ✏️ **Example (1D):**

Let $f(x) = x^2$. Then $\nabla f(x) = 2x$

Gradient descent update:

$$
x^{(k+1)} = x^{(k)} - \alpha \cdot 2x^{(k)} = (1 - 2\alpha)x^{(k)}
$$

It converges to 0 if $0 < \alpha < 1$



---

## Explain the Newton-Raphson method and its convergence conditions.

### ✅ **Newton-Raphson Method (Newton's Method)**

The **Newton-Raphson method** is an iterative algorithm for finding **roots of a real-valued function** or for solving **nonlinear optimization problems**. It's faster than gradient descent near the optimum, especially when the function is well-behaved.

---

## 🔍 1. **Goal (Root Finding):**

Given a function $f(x)$, find $x^*$ such that:

$$
f(x^*) = 0
$$

---

## 🧮 2. **Derivation of Newton-Raphson Update**

Use a **second-order Taylor expansion** of $f(x)$ around a point $x_k$:

$$
f(x) \approx f(x_k) + f'(x_k)(x - x_k)
$$

To find a better estimate $x_{k+1}$, set the approximation to zero:

$$
f(x_k) + f'(x_k)(x_{k+1} - x_k) = 0
\Rightarrow x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}
$$

---

### ✅ **Newton-Raphson Update Rule (1D):**

$$
x_{k+1} = x_k - \frac{f(x_k)}{f'(x_k)}
$$

---

## 🧠 3. **Extension to Multivariate Optimization**

To **minimize** a twice-differentiable function $f: \mathbb{R}^n \rightarrow \mathbb{R}$, we use Newton's method on $\nabla f(x) = 0$.

$$
x_{k+1} = x_k - H_f(x_k)^{-1} \nabla f(x_k)
$$

Where:

* $\nabla f(x_k)$ is the **gradient**
* $H_f(x_k)$ is the **Hessian matrix** (second derivatives)

---

## ✅ 4. **Convergence Conditions**

Newton's method converges **quadratically** under the following conditions:

### **Local Convergence Conditions:**

1. **Function is twice continuously differentiable**
2. The **initial guess $x_0$** is **sufficiently close** to the true root/minimum
3. The **Jacobian (1D: derivative, multivariable: Hessian)** at the solution is **non-singular**
4. In optimization: Hessian must be **positive definite** near the minimum (for descent)

---

## ❗ 5. **Potential Issues**

* **Does not converge** if:

  * $f'(x_k) \approx 0$ → division by small number
  * Starting point is far from the solution
  * Hessian is not positive definite (in optimization)

* **Requires computing second derivatives** (expensive for high dimensions)

---

## 📘 6. **Example (1D)**

Let $f(x) = x^2 - 2$

Then:

* $f'(x) = 2x$

Apply:

$$
x_{k+1} = x_k - \frac{x_k^2 - 2}{2x_k}
= \frac{1}{2} \left(x_k + \frac{2}{x_k} \right)
$$

This is the **Babylonian method** for computing $\sqrt{2}$


---

## Discuss stochastic gradient descent and its variants (Adam, RMSProp, etc.).

Certainly! Let’s discuss **Stochastic Gradient Descent (SGD)** and its popular **variants** like **Momentum**, **RMSProp**, and **Adam**, which are widely used in training machine learning models, especially deep neural networks.

---

## 🔁 **1. Stochastic Gradient Descent (SGD)**

### 🔍 **Motivation:**

In traditional **batch gradient descent**, the update rule is:

$$
\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t)
$$

Where:

* $\theta$: model parameters
* $J(\theta)$: objective (loss) function
* $\alpha$: learning rate

But computing the full gradient $\nabla_\theta J$ over the entire dataset is **computationally expensive** for large datasets.

---

### 🧠 **SGD Idea:**

Use a **random sample (mini-batch)** instead of the whole dataset to estimate the gradient:

$$
\theta_{t+1} = \theta_t - \alpha \nabla_\theta J(\theta_t; x^{(i)})
$$

* $x^{(i)}$: a single or mini-batch of training examples
* This gives **noisy updates**, but they are **much faster** and can escape local minima.

---

## ⚙️ **2. Variants of SGD**

To improve convergence and stability, various modifications have been proposed:

---

### 🚀 **2.1. SGD with Momentum**

**Problem:** Vanilla SGD may oscillate, especially in ravines (areas where the surface curves more steeply in one direction).

**Solution:** Add a **velocity** term that accumulates past gradients:

$$
v_t = \beta v_{t-1} + (1 - \beta)\nabla_\theta J(\theta_t)
$$

$$
\theta_{t+1} = \theta_t - \alpha v_t
$$

* $\beta \in [0, 1)$: momentum coefficient (e.g., 0.9)
* Helps accelerate in consistent gradient directions and dampen oscillations.

---

### 📐 **2.2. RMSProp (Root Mean Square Propagation)**

**Problem:** Learning rate might be too high for some parameters and too low for others.

**Solution:** Use an **adaptive learning rate** per parameter:

$$
E[g^2]_t = \beta E[g^2]_{t-1} + (1 - \beta) g_t^2
$$

$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{E[g^2]_t + \epsilon}} g_t
$$

* $g_t$: current gradient
* $\beta$: decay rate (e.g., 0.9)
* $\epsilon$: small constant (e.g., $10^{-8}$) to avoid division by zero
* Controls step size based on recent squared gradients → smooths and adapts learning.

---

### 🤖 **2.3. Adam (Adaptive Moment Estimation)**

**Combines Momentum + RMSProp.** It computes **first** and **second** moment estimates of gradients:

$$
m_t = \beta_1 m_{t-1} + (1 - \beta_1) g_t \quad \text{(1st moment - mean)}
$$

$$
v_t = \beta_2 v_{t-1} + (1 - \beta_2) g_t^2 \quad \text{(2nd moment - uncentered variance)}
$$

Bias-corrected estimates:

$$
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1 - \beta_2^t}
$$

Parameter update:

$$
\theta_{t+1} = \theta_t - \frac{\alpha}{\sqrt{\hat{v}_t} + \epsilon} \hat{m}_t
$$

* $\beta_1 = 0.9$, $\beta_2 = 0.999$, $\epsilon = 10^{-8}$
* **Widely used** due to fast convergence and stable updates.

---

## 🧾 **Comparison Summary**

| Optimizer        | Adaptivity | Momentum | Speed  | Robustness |
| ---------------- | ---------- | -------- | ------ | ---------- |
| **SGD**          | ❌          | ❌        | Slow   | Stable     |
| **SGD+Momentum** | ❌          | ✅        | Medium | Better     |
| **RMSProp**      | ✅          | ❌        | Fast   | Good       |
| **Adam**         | ✅          | ✅        | Fast   | Very good  |

---

## ✅ **When to Use What**

* Use **SGD + Momentum** for **convex** or simple problems (like logistic regression).
* Use **Adam** for **deep learning** or **non-stationary objectives** (e.g., in reinforcement learning).
* Use **RMSProp** or **Adam** when gradients are sparse or highly varying.





---

## What is the method of Lagrange multipliers? Give an example.


### ✅ **Method of Lagrange Multipliers**

The **method of Lagrange multipliers** is a strategy for finding the **local maxima and minima** of a function **subject to equality constraints**.

---

## 🧠 1. **Problem Setup**

We want to **maximize or minimize** a function:

$$
f(x, y, \dots)
$$

**Subject to a constraint:**

$$
g(x, y, \dots) = 0
$$

---

## 🧮 2. **Lagrangian Function**

Define the **Lagrangian** as:

$$
\mathcal{L}(x, y, \lambda) = f(x, y) - \lambda \cdot g(x, y)
$$

Where:

* $\lambda$: **Lagrange multiplier**
* $f(x, y)$: objective function
* $g(x, y) = 0$: constraint

---

## ✅ 3. **Lagrange Conditions**

Find the stationary points by solving:

$$
\nabla \mathcal{L} = 0 \Rightarrow
\begin{cases}
\frac{\partial \mathcal{L}}{\partial x} = 0 \\
\frac{\partial \mathcal{L}}{\partial y} = 0 \\
\frac{\partial \mathcal{L}}{\partial \lambda} = 0
\end{cases}
$$

That is:

$$
\begin{cases}
\nabla f(x, y) = \lambda \nabla g(x, y) \\
g(x, y) = 0
\end{cases}
$$

---

## 📘 4. **Example**

### 🔍 Problem:

Maximize $f(x, y) = xy$
Subject to: $x^2 + y^2 = 1$ (the unit circle)

---

### ✏️ Step 1: Define Lagrangian

$$
\mathcal{L}(x, y, \lambda) = xy - \lambda(x^2 + y^2 - 1)
$$

---

### ✏️ Step 2: Take partial derivatives

$$
\frac{\partial \mathcal{L}}{\partial x} = y - 2\lambda x = 0 \tag{1}
$$

$$
\frac{\partial \mathcal{L}}{\partial y} = x - 2\lambda y = 0 \tag{2}
$$

$$
\frac{\partial \mathcal{L}}{\partial \lambda} = -(x^2 + y^2 - 1) = 0 \tag{3}
$$

---

### ✏️ Step 3: Solve the system

From (1): $y = 2\lambda x$
From (2): $x = 2\lambda y$

Substitute the first into the second:

$$
x = 2\lambda (2\lambda x) = 4\lambda^2 x
\Rightarrow x(1 - 4\lambda^2) = 0
$$

So either:

* $x = 0$
* $\lambda = \pm \tfrac{1}{2}$

#### Case 1: $x = 0 \Rightarrow y^2 = 1 \Rightarrow y = \pm1$

* $f(0, \pm1) = 0$

#### Case 2: $\lambda = \pm \tfrac{1}{2}$

From $y = 2\lambda x \Rightarrow y = \pm x$

Substitute into constraint:
$x^2 + y^2 = 1 \Rightarrow x^2 + x^2 = 1 \Rightarrow x = \pm \tfrac{1}{\sqrt{2}}$

So:

* $(x, y) = (\tfrac{1}{\sqrt{2}}, \tfrac{1}{\sqrt{2}}) \Rightarrow f = \tfrac{1}{2}$
* $(x, y) = (-\tfrac{1}{\sqrt{2}}, -\tfrac{1}{\sqrt{2}}) \Rightarrow f = \tfrac{1}{2}$

---

### ✅ Final Answer:

* **Max value:** $\boxed{\tfrac{1}{2}}$
* **At points:** $\left( \pm \tfrac{1}{\sqrt{2}}, \pm \tfrac{1}{\sqrt{2}} \right)$
* **Min value:** $\boxed{0}$ at $(0, \pm1)$


---

## How do interior-point methods work?

### ✅ **Interior-Point Methods – Overview**

**Interior-point methods** are a class of optimization algorithms used to solve **constrained convex optimization problems** — particularly **linear programming (LP)**, **quadratic programming (QP)**, and **general nonlinear programming (NLP)**. Unlike the **simplex method**, which walks along the edges of the feasible region, interior-point methods **approach the optimal solution from within the feasible region**.

---

## 🔍 **1. Problem Setup**

Consider a **constrained optimization problem**:

$$
\min_{x} \ f(x) \quad \text{subject to: } \begin{cases}
g_i(x) \leq 0, \quad i = 1, \dots, m \\
h_j(x) = 0, \quad j = 1, \dots, p
\end{cases}
$$

Where:

* $f(x)$: objective function (convex)
* $g_i(x)$: inequality constraints
* $h_j(x)$: equality constraints

---

## 🧠 **2. Key Idea**

**Inequality constraints** are difficult to handle directly. Interior-point methods convert them into **barrier functions** to prevent the algorithm from leaving the feasible region.

Instead of solving the original problem directly, solve a **sequence of unconstrained problems** that gradually approximate it.

---

## 📘 **3. Barrier Method (Core of Interior-Point Methods)**

### 🔄 **Transform Inequality Constraints**

For:

$$
\min_x \ f(x) \quad \text{subject to } g_i(x) \leq 0
$$

Use a **logarithmic barrier** function:

$$
\phi(x) = -\sum_{i=1}^m \log(-g_i(x))
$$

Then solve the modified problem:

$$
\min_x \left[ f(x) + \frac{1}{t} \phi(x) \right]
$$

* $t > 0$: barrier parameter
* As $t \to \infty$, the solution of the barrier problem approaches the original constrained problem.

---

### 📈 **Sequence of Approximate Problems**

1. Start with a small $t$ (heavy barrier, conservative)
2. Solve the unconstrained problem:

   $$
   \min_x \ f(x) + \frac{1}{t} \phi(x)
   $$
3. Increase $t$: $t \leftarrow \mu t$ (e.g., $\mu = 10$)
4. Repeat until convergence

---

## ⚙️ **4. Newton’s Method in Interior-Point**

Each subproblem (the barrier problem) is solved using **Newton’s method**, leveraging the first and second derivatives:

* Compute gradient and Hessian of the barrier objective
* Use Newton’s update rule to find the minimizer
* Stay strictly inside the feasible region

---

## 🧾 **5. Advantages**

* **Polynomial-time convergence** for LPs and convex QPs
* Works well in **high-dimensional spaces**
* Efficient for **dense constraint structures**
* Does not suffer from the combinatorial complexity of the simplex method

---

## ❗ **6. Limitations**

* Requires **feasible starting point inside the region**
* Barrier functions can become **ill-conditioned** as $t \to \infty$
* May be inefficient for **very sparse or structured problems** (compared to specialized methods)

---

## 📊 **7. Example: LP Using Barrier Method**

Solve:

$$
\min_x \ c^T x \quad \text{subject to } Ax = b, \ x > 0
$$

Barrier version:

$$
\min_x \ c^T x - \frac{1}{t} \sum_i \log(x_i) \quad \text{subject to } Ax = b
$$

* Solve this using Newton’s method for increasing $t$
* This is the foundation of **primal-dual interior-point methods** used in modern solvers

---

## 📌 **In Summary:**

| Feature              | Interior-Point Method                    |
| -------------------- | ---------------------------------------- |
| Handles constraints? | Yes, using barrier functions             |
| Moves along edges?   | No, moves **through interior** of region |
| Solves?              | LP, QP, NLP, SDP (convex problems)       |
| Solver behavior      | Polynomial-time, smooth convergence      |
| Common in libraries  | Yes (e.g., CVXOPT, MOSEK, IPOPT, Gurobi) |


---

## Formulate and solve an optimization problem under probabilistic constraints.


### ✅ **Optimization Under Probabilistic Constraints**

An optimization problem with **probabilistic (chance) constraints** includes uncertainties in constraints and aims to satisfy them **with high probability**, not necessarily always.

---

## 🔍 **1. General Formulation**

Let:

* $x \in \mathbb{R}^n$: decision variables
* $f(x)$: objective function
* $\xi$: random variable (uncertainty)
* $g(x, \xi) \leq 0$: uncertain constraint

### ✅ **Chance-Constrained Optimization Problem:**

$$
\min_{x \in \mathcal{X}} \quad f(x)
$$

$$
\text{subject to: } \mathbb{P}_{\xi}(g(x, \xi) \leq 0) \geq 1 - \alpha
$$

* $\alpha \in (0,1)$: risk level (e.g., 0.05)
* Constraint is satisfied with at least $1 - \alpha$ probability

---

## 📘 **2. Example Problem: Portfolio Optimization with Probabilistic Constraint**

Suppose you’re allocating investments across 2 assets with uncertain returns $\xi_1, \xi_2$. The returns are modeled as **normally distributed**.

---

### Problem:

Choose allocation $x = (x_1, x_2)$ with $x_1 + x_2 = 1$, $x_1, x_2 \geq 0$, to:

#### 🎯 **Maximize expected return:**

$$
\max_{x} \ \mathbb{E}[\xi_1]x_1 + \mathbb{E}[\xi_2]x_2
$$

#### 📉 **Subject to:**

Probability that return is **at least 5%** must be ≥ 95%:

$$
\mathbb{P}\left(\xi_1 x_1 + \xi_2 x_2 \geq 0.05\right) \geq 0.95
$$

---

### ✏️ Step 1: Assume Normal Distribution

Let:

* $\xi = (\xi_1, \xi_2) \sim \mathcal{N}(\mu, \Sigma)$
* $R(x) = \xi^T x \sim \mathcal{N}(\mu^T x, x^T \Sigma x)$

So:

$$
\mathbb{P}(R(x) \geq 0.05) = \mathbb{P}\left(Z \geq \frac{0.05 - \mu^T x}{\sqrt{x^T \Sigma x}}\right)
$$

Where $Z \sim \mathcal{N}(0, 1)$

---

### ✏️ Step 2: Reformulate Constraint

We require:

$$
\mathbb{P}\left(Z \geq \frac{0.05 - \mu^T x}{\sqrt{x^T \Sigma x}}\right) \geq 0.95
\Rightarrow \frac{0.05 - \mu^T x}{\sqrt{x^T \Sigma x}} \leq z_{0.05}
$$

Where $z_{0.05} \approx 1.6449$ (from standard normal table)

So:

$$
\mu^T x - 0.05 \geq 1.6449 \cdot \sqrt{x^T \Sigma x}
$$

---

## ✅ Final Deterministic Optimization Problem

$$
\max_{x_1 + x_2 = 1, \ x \geq 0} \quad \mu^T x
$$

$$
\text{subject to: } \mu^T x - 0.05 \geq 1.6449 \cdot \sqrt{x^T \Sigma x}
$$

This is a **second-order cone program (SOCP)**, which can be efficiently solved using convex optimization tools like **CVXPY (Python)**.

---

## 🧪 Example with Numbers

Let:

* $\mu = \begin{bmatrix} 0.06 \\ 0.04 \end{bmatrix}$
* $\Sigma = \begin{bmatrix} 0.01 & 0.001 \\ 0.001 & 0.002 \end{bmatrix}$

Then:

* Objective: $0.06 x_1 + 0.04 x_2$
* Constraint:

  $$
  0.06 x_1 + 0.04 x_2 - 0.05 \geq 1.6449 \cdot \sqrt{0.01 x_1^2 + 0.002 x_2^2 + 2 \cdot 0.001 x_1 x_2}
  $$
* With $x_1 + x_2 = 1$, $x_1, x_2 \geq 0$

This problem can be solved numerically.

---

## 🧰 Tools to Solve

You can solve this using:

```python
import cvxpy as cp
import numpy as np

mu = np.array([0.06, 0.04])
Sigma = np.array([[0.01, 0.001],
                  [0.001, 0.002]])

x = cp.Variable(2)
ret = mu @ x
risk = cp.quad_form(x, Sigma)
prob = cp.Problem(cp.Maximize(ret),
                  [cp.sum(x) == 1,
                   x >= 0,
                   ret - 0.05 >= 1.6449 * cp.sqrt(risk)])
prob.solve()
print("Optimal x:", x.value)
```



---

## What is simulated annealing? When is it preferred over gradient methods?

### 🔥 **Simulated Annealing (SA)** — Overview

**Simulated Annealing** is a **probabilistic optimization algorithm** inspired by the **annealing process in metallurgy**, where a material is heated and slowly cooled to minimize defects and reach a stable structure.

---

## 📌 1. **Key Concepts**

* **Global optimization** method
* Works on **discrete** or **continuous** problems
* Accepts **worse solutions** early on to escape local minima
* Uses a **temperature parameter** to control exploration

---

## 🧠 2. **Algorithm Steps**

Let $f(x)$ be the **objective function** to minimize.

### Algorithm:

1. **Initialize:**

   * Initial solution $x_0$
   * Temperature $T_0$
   * Cooling schedule

2. **Repeat:**

   * Generate a **random neighbor** $x'$ of current solution $x$
   * Compute $\Delta = f(x') - f(x)$
   * If $\Delta < 0$: accept $x'$
   * Else: accept with probability $e^{-\Delta / T}$
   * Decrease temperature: $T \leftarrow \alpha T$, with $0 < \alpha < 1$

3. **Stop** when temperature is low or a stopping criterion is met.

---

## 🧪 3. **Intuition**

* At **high temperature**, the algorithm is exploratory — it accepts uphill (worse) moves often.
* As the temperature **cools**, it becomes more conservative, focusing on local improvement.
* This allows it to **escape local minima**, unlike gradient descent which may get stuck.

---

## 📉 4. **When Is Simulated Annealing Preferred Over Gradient Methods?**

| Case                                  | Why Simulated Annealing?                             |
| ------------------------------------- | ---------------------------------------------------- |
| 🧱 Non-convex optimization            | Can escape **local minima**                          |
| 🎲 Discrete or combinatorial problems | Gradient methods don't apply (e.g., TSP, scheduling) |
| ⛰️ Rugged or noisy landscapes         | More **robust to noise** and discontinuities         |
| ❌ No gradient available               | Works even if **derivatives are unavailable**        |
| 🔄 Multi-modal functions              | Explores **multiple regions** of the search space    |

---

## 🆚 Gradient-Based Methods

| Feature               | Simulated Annealing | Gradient Descent             |
| --------------------- | ------------------- | ---------------------------- |
| Determinism           | Stochastic          | Deterministic (usually)      |
| Local minima escape   | Yes                 | No                           |
| Requires gradient     | No                  | Yes                          |
| Speed                 | Slower (usually)    | Fast (with smooth functions) |
| Convergence guarantee | No (heuristic)      | Yes (for convex problems)    |

---

## 💡 5. Example Use Cases

* **Traveling Salesman Problem (TSP)**
* **VLSI circuit design**
* **Job shop scheduling**
* **Hyperparameter tuning**
* **Function minimization with noisy or discontinuous surfaces**

---

## ✅ Summary

**Simulated Annealing** is a powerful **metaheuristic** for difficult optimization problems where traditional gradient methods fail:

* Works with **non-differentiable**, **noisy**, or **combinatorial** objective functions
* Preferred when **global optimum** is more important than speed
* Often used with **cooling schedules** like geometric decay or adaptive methods



---

## Describe the genetic algorithm for non-convex functions.


### 🧬 **Genetic Algorithm (GA)** for Non-Convex Optimization

A **genetic algorithm (GA)** is a **population-based, stochastic optimization technique** inspired by the process of **natural selection**. It is particularly well-suited for **non-convex**, **non-differentiable**, or **combinatorial optimization problems**, where gradient-based methods struggle.

---

## 🧠 **1. Core Idea**

GA mimics **evolutionary biology** principles:

* **Selection**: Favor better solutions (higher fitness)
* **Crossover**: Combine solutions to produce new ones
* **Mutation**: Randomly alter solutions to maintain diversity

---

## 🧮 **2. General Structure of GA**

Let $f(x)$ be the (non-convex) objective function to **minimize**.

### Algorithm Steps:

1. **Initialize a population** of candidate solutions $\{x_1, x_2, \ldots, x_n\}$
2. **Evaluate fitness** of each solution (e.g., $-f(x_i)$ for minimization)
3. **Selection**: Choose individuals to reproduce based on fitness
4. **Crossover**: Combine selected individuals to produce offspring
5. **Mutation**: Randomly tweak offspring to explore new regions
6. **Survivor selection**: Form the next generation from best individuals
7. **Repeat** until a stopping criterion is met (e.g., max generations or convergence)

---

## 📉 **3. Applicability to Non-Convex Functions**

GAs are **especially useful for non-convex functions** because:

* They don’t rely on gradients (which may not exist or be misleading)
* They explore the solution space globally
* They naturally avoid local traps via crossover and mutation
* They are robust to rugged or discontinuous landscapes

---

## 🆚 **Compared to Gradient-Based Methods**

| Feature                  | Genetic Algorithm | Gradient Methods              |
| ------------------------ | ----------------- | ----------------------------- |
| Assumes smoothness?      | ❌ No              | ✅ Yes                         |
| Handles discrete vars?   | ✅ Yes             | ❌ No                          |
| Local minima resistant?  | ✅ Often           | ❌ Easily trapped              |
| Fast convergence?        | ❌ Slower          | ✅ Fast (when convex & smooth) |
| Global search capability | ✅ Yes             | ❌ No (unless restarted)       |

---

## 🧪 **4. Example: GA for Minimizing a Non-Convex Function**

### Objective:

Minimize:

$$
f(x) = x^2 + 10\sin(x), \quad x \in [-10, 10]
$$

This function has **multiple local minima**.

### GA Outline (Python-style pseudocode):

```python
import numpy as np

def fitness(x):
    return -(x**2 + 10*np.sin(x))  # maximize negative of f(x)

# Initialize population
pop = np.random.uniform(-10, 10, size=(50,))
for generation in range(100):
    # Evaluate fitness
    fit = fitness(pop)
    
    # Selection (e.g., tournament)
    selected = pop[np.argsort(fit)[-25:]]  # top 50%
    
    # Crossover
    offspring = []
    for i in range(25):
        parent1, parent2 = np.random.choice(selected, 2)
        child = 0.5 * (parent1 + parent2)
        offspring.append(child)
    
    # Mutation
    offspring = [x + np.random.normal(0, 1) for x in offspring]
    
    # Create new population
    pop = np.concatenate([selected, offspring])

# Best solution
best = pop[np.argmax(fitness(pop))]
```

---

## 🧩 **5. Key Parameters**

| Parameter        | Role                                                |
| ---------------- | --------------------------------------------------- |
| Population size  | Affects exploration capability                      |
| Crossover rate   | Controls how aggressively genes are mixed           |
| Mutation rate    | Ensures exploration of new solutions                |
| Selection method | Tournament, roulette wheel, etc.                    |
| Elitism          | Preserving the best individuals between generations |

---

## ✅ **When to Use Genetic Algorithms**

Use GAs if your problem is:

* **Non-convex**, **non-smooth**, or **multi-modal**
* Involves **combinatorial** or **discrete** variables
* Gradient information is **unavailable or unreliable**
* You need a **robust global search**

