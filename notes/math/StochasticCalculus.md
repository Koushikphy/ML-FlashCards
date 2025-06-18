## Standard Brownian Motion 

(also called Wiener Process) is a continuous-time stochastic process that serves as a mathematical model for random motion. It is a cornerstone in fields like quantitative finance, physics, and stochastic calculus.

### **Definition**

A standard Brownian motion $\{W_t\}_{t \geq 0}$ is a stochastic process with the following key properties:

---

### ✅ **Key Properties:**

1. **Initial Value**

$$
   W_0 = 0
$$

   The process starts at zero.

2. **Independent Increments**

$$
   W_{t_2} - W_{t_1}, W_{t_3} - W_{t_2}, \dots \text{ are independent for } t_1 < t_2 < t_3 < \dots
$$

   This means the future evolution of the process is independent of the past.

3. **Stationary and Normally Distributed Increments**

$$
   W_{t+s} - W_t \sim \mathcal{N}(0, s)
$$

   The increments are normally distributed with **mean 0** and **variance equal to the length of the time interval**.

4. **Continuity of Paths**
   The sample paths $t \mapsto W_t$ are continuous almost surely (no jumps), though they are nowhere differentiable.

5. **Martingale Property**
$\mathbb{E}[W_t \mid \mathcal{F}_s] = W_s$ for $s \le t$, meaning Brownian motion has no drift — its best prediction at future time $t$ is just its current value $W_s$.

6. **Markov Property**
   The future of the process depends only on the present, not the past: $W_t$ is a **Markov process**.

---

### 🔍 **Other Notes:**

* The variance of $W_t$ grows linearly with time: $\text{Var}(W_t) = t$.
* Brownian motion is a **Gaussian process** — every finite collection of values has a joint multivariate normal distribution.





## ✅ **Geometric Brownian Motion (GBM)**

**Definition:**
Geometric Brownian Motion is a continuous-time stochastic process used to model variables that evolve **multiplicatively**, such as stock prices. It is defined as:

$$
S_t = S_0 \exp\left( \left(\mu - \frac{1}{2} \sigma^2\right)t + \sigma W_t \right)
$$

Where:

* $S_t$: value of the process at time $t$
* $S_0$: initial value (e.g., initial stock price)
* $\mu$: drift term (mean growth rate)
* $\sigma$: volatility (standard deviation)
* $W_t$: standard Brownian motion

---

### 🔁 **How GBM Differs from Standard Brownian Motion**

| Feature                         | **Standard Brownian Motion $W_t$** | **Geometric Brownian Motion $S_t$**                        |
| ------------------------------- | ---------------------------------- | ---------------------------------------------------------- |
| **Equation**                    | $W_t$                              | $S_t = S_0 e^{(\mu - \frac{1}{2} \sigma^2)t + \sigma W_t}$ |
| **Range of Values**             | Real line $(-\infty, \infty)$      | Positive real numbers $(0, \infty)$                        |
| **Additive vs. Multiplicative** | Additive process                   | Multiplicative (exponential of Brownian motion)            |
| **Stationarity of increments**  | Yes (increments are stationary)    | No (variance grows faster due to exponential scaling)      |
| **Mean**                        | $\mathbb{E}[W_t] = 0$              | $\mathbb{E}[S_t] = S_0 e^{\mu t}$                          |
| **Variance**                    | $\text{Var}(W_t) = t$              | $\text{Var}(S_t) = S_0^2 e^{2\mu t}(e^{\sigma^2 t} - 1)$   |

---

### 📌 **Why GBM Is Used in Finance:**

* **No negative values:** Stock prices can’t go negative — GBM ensures positive values.
* **Log-normal distribution:** Since $\log S_t$ is normally distributed, $S_t$ is log-normally distributed — matching observed market behavior.
* **Captures compounding:** Reflects real-world behavior of growth processes like interest, inflation, or investment returns.



In short, **GBM is an exponential transformation of Brownian motion**, introducing drift and ensuring positivity, making it ideal for modeling prices and other quantities that evolve over time without going negative.

---

## ✅ **Quadratic Variation – Definition**

The **quadratic variation** of a stochastic process $X_t$ over the interval $[0, t]$ is defined as the limit of the sum of squared increments as the partition gets finer:

$$
[X]_t = \lim_{|\Delta| \to 0} \sum_{i=0}^{n-1} (X_{t_{i+1}} - X_{t_i})^2
$$

where $\{t_i\}$ is a partition of $[0, t]$, and $|\Delta|$ is the mesh (maximum width) of the partition.



### 📌 **Quadratic Variation of Standard Brownian Motion**

Let $W_t$ be standard Brownian motion. Then:

$$
[W]_t = t
$$

This means the quadratic variation of Brownian motion **increases linearly** with time, even though Brownian paths are continuous and nowhere differentiable.



### 🔍 **Why Is This Important?**

1. **Brownian Motion Has Infinite Total Variation**
   While Brownian motion is continuous, it is highly irregular. Its path has **infinite total variation**, but its **quadratic variation is finite** and deterministic: $t$. This is a key signature of stochastic processes with noise.

2. **Used in Ito Calculus**
   Quadratic variation is fundamental to **Ito’s lemma**. For example, the differential of $W_t^2$ is:

$$
   d(W_t^2) = 2W_t dW_t + dt
$$

   That extra $dt$ term comes from the **quadratic variation**: $(dW_t)^2 = dt$, which is **non-zero**, unlike in classical calculus.

3. **Helps Identify Brownian Motion**
   A continuous martingale $M_t$ with $[M]_t = t$ can often be shown to be Brownian motion via Lévy’s characterization.



### 🧠 **Intuition**

Even though the average displacement of Brownian motion is zero, the **accumulated "wiggliness"** (squared displacements) grows steadily — that's what quadratic variation captures.



In short, **quadratic variation measures the accumulated volatility or randomness** of a process — and for Brownian motion, it grows linearly with time:

$$
[W]_t = t
$$

---


## Derive **Ito's Lemma** for a function $f(t, X_t)$, where $X_t$ is an **Itô process**.



### 🔷 **Step 1: Define the Itô Process**

Let $X_t$ be defined as:

$$
dX_t = \mu(t, X_t) dt + \sigma(t, X_t) dW_t
$$

Where:

* $\mu(t, X_t)$: drift term
* $\sigma(t, X_t)$: diffusion term
* $W_t$: standard Brownian motion



### 🔷 **Step 2: Goal – Find $df(t, X_t)$**

We want to find the differential of $f(t, X_t)$, a twice continuously differentiable function:

$$
f: \mathbb{R}^+ \times \mathbb{R} \to \mathbb{R}
$$



### 🔷 **Step 3: Use Multivariate Taylor Expansion (Stochastic Version)**

The **Itô version of Taylor expansion** includes second-order terms, due to the nonzero quadratic variation of Brownian motion:

$$
df(t, X_t) = \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} dX_t + \frac{1}{2} \frac{\partial^2 f}{\partial x^2} (dX_t)^2
$$

Now plug in $dX_t = \mu dt + \sigma dW_t$:

* $dX_t = \mu dt + \sigma dW_t$
* $(dX_t)^2 = \sigma^2 dt$ since $(dW_t)^2 = dt$, and cross terms like $dt \cdot dW_t$ vanish in Itô calculus



### 🔷 **Step 4: Substitute and Simplify**

$$
\begin{aligned}
df(t, X_t) &= \frac{\partial f}{\partial t} dt + \frac{\partial f}{\partial x} (\mu dt + \sigma dW_t) + \frac{1}{2} \frac{\partial^2 f}{\partial x^2} \sigma^2 dt \\
&= \left( \frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2} \sigma^2 \frac{\partial^2 f}{\partial x^2} \right) dt + \sigma \frac{\partial f}{\partial x} dW_t
\end{aligned}
$$



### ✅ **Final Result — Itô's Lemma:**

$$
\boxed{
df(t, X_t) = \left( \frac{\partial f}{\partial t} + \mu \frac{\partial f}{\partial x} + \frac{1}{2} \sigma^2 \frac{\partial^2 f}{\partial x^2} \right) dt + \sigma \frac{\partial f}{\partial x} dW_t
}
$$

This is the **single-variable version** of Itô’s Lemma.



### 📌 **Applications:**

* Crucial in pricing derivatives (e.g., Black-Scholes)
* Foundation for solving stochastic differential equations (SDEs)
* Allows transformation of stochastic processes through functions (change of variables)

---

## Apply Ito’s Lemma to $\ln(S_t)$ where $dS_t = \mu S_t dt + \sigma S_t dW_t$.


Let's apply **Itô’s Lemma** to the function:

$$
f(S_t) = \ln(S_t)
$$

Where $S_t$ follows a **geometric Brownian motion**:

$$
dS_t = \mu S_t   dt + \sigma S_t   dW_t
$$



### 🔷 Step 1: Identify the components

This is a function of one stochastic variable $S_t$, and:

* $\mu(S_t) = \mu S_t$
* $\sigma(S_t) = \sigma S_t$
* $f(S_t) = \ln(S_t)$

Compute the derivatives of $f(S_t)$:

* $f'(S_t) = \frac{1}{S_t}$
* $f''(S_t) = -\frac{1}{S_t^2}$



### 🔷 Step 2: Use Itô’s Lemma for functions of one variable:

$$
df(S_t) = \left( \frac{\partial f}{\partial t} + \mu(S_t) \frac{\partial f}{\partial S} + \frac{1}{2} \sigma(S_t)^2 \frac{\partial^2 f}{\partial S^2} \right) dt + \sigma(S_t) \frac{\partial f}{\partial S} dW_t
$$

Since $f$ has no explicit time dependence ($\partial f/\partial t = 0$), we simplify:

$$
d\ln(S_t) = \left( \mu S_t \cdot \frac{1}{S_t} + \frac{1}{2} \sigma^2 S_t^2 \cdot \left(-\frac{1}{S_t^2}\right) \right) dt + \sigma S_t \cdot \frac{1}{S_t} dW_t
$$



### 🔷 Step 3: Simplify terms

$$
d\ln(S_t) = \left( \mu - \frac{1}{2} \sigma^2 \right) dt + \sigma   dW_t
$$



### ✅ **Final Result:**

$$
\boxed{d\ln(S_t) = \left( \mu - \frac{1}{2} \sigma^2 \right) dt + \sigma   dW_t}
$$



### 🔍 Interpretation:

* The **logarithmic return** $\ln(S_t)$ follows a linear Brownian motion with drift $\mu - \frac{1}{2}\sigma^2$ and volatility $\sigma$.
* This is why **log returns of GBM are normally distributed**, a key assumption in the Black-Scholes model.



---

## Explain the difference between Ito and Stratonovich integrals.

The **Itô and Stratonovich integrals** are two different ways to define integrals with respect to **stochastic processes**, particularly Brownian motion. Though they may look similar, they yield different results and have different mathematical properties and interpretations.



## ✅ **1. Setup: Stochastic Integral**

We want to define an integral of the form:

$$
\int_0^t f(X_s)   dW_s
$$

Where $W_s$ is Brownian motion and $f(X_s)$ is some adapted (random) process. The question is: **how do we define the value of the integrand inside each time increment?**




### 🔹 **2. Itô Integral**

### **Definition:**

In the **Itô integral**, the integrand is evaluated at the **left endpoint** of each subinterval:

$$
\int_0^t f(X_s)   dW_s \quad \text{(Itô)} \quad \approx \sum f(X_{t_i}) \cdot (W_{t_{i+1}} - W_{t_i})
$$

### **Key Properties:**

| Feature             | Itô Integral                                                 |
| ------------------- | ------------------------------------------------------------ |
| Evaluation point    | **Left endpoint** $t_i$                                      |
| Bias                | **Introduces a correction term** in chain rule (Itô's lemma) |
| Quadratic variation | Takes $(dW_t)^2 = dt$ explicitly into account                |
| Martingale property | The integral is a **martingale**                             |
| Common in           | **Finance**, stochastic calculus, SDE theory                 |
| Chain rule          | Non-standard: includes second-order term (Itô's Lemma)       |

---

### 🔹 **3. Stratonovich Integral**

### **Definition:**

In the **Stratonovich integral**, the integrand is evaluated at the **midpoint** (or average of endpoints):

$$
\int_0^t f(X_s) \circ dW_s \quad \text{(Stratonovich)} \quad \approx \sum f\left( \frac{X_{t_i} + X_{t_{i+1}}}{2} \right) \cdot (W_{t_{i+1}} - W_{t_i})
$$

Here, the $\circ dW_s$ notation indicates the **Stratonovich** integral.

### **Key Properties:**

| Feature             | Stratonovich Integral                                                |
| ------------------- | -------------------------------------------------------------------- |
| Evaluation point    | **Midpoint** or average                                              |
| Chain rule          | **Classical chain rule applies** (no correction term)                |
| Quadratic variation | Implicitly smooths it out                                            |
| Martingale property | **Not necessarily a martingale**                                     |
| Common in           | **Physics**, engineering, systems with physical noise                |
| Good for            | Systems derived as limits of regular processes (e.g., colored noise) |

---

### 🔁 **4. Relationship Between Itô and Stratonovich**

They are related by:

$$
\int_0^t f(X_s) \circ dW_s = \int_0^t f(X_s)   dW_s + \frac{1}{2} \int_0^t f'(X_s) \sigma(X_s)   ds
$$

That extra term accounts for the difference due to the **midpoint vs. left-point evaluation**.



### 🎯 Summary Table:

| Feature               | Itô                    | Stratonovich                            |
| --------------------- | ---------------------- | --------------------------------------- |
| Evaluation point      | Left endpoint          | Midpoint (average)                      |
| Chain rule            | Modified (Itô’s lemma) | Standard (classical calculus)           |
| Martingale property   | Yes                    | No                                      |
| Use case (common)     | Finance, SDEs          | Physics, engineering                    |
| Extra correction term | Yes                    | No (but equivalent to Itô + correction) |
| Intuition             | "Future is unknown"    | "Smoothed noise"                        |

---


## Solve the SDE: $dX_t = \mu X_t dt + \sigma X_t dW_t$.

We are given the **stochastic differential equation (SDE)**:

$$
dX_t = \mu X_t  dt + \sigma X_t  dW_t,
$$

where:

* $\mu$ is the **drift coefficient**,
* $\sigma$ is the **diffusion coefficient**,
* $W_t$ is a **standard Brownian motion**,
* $X_t$ is the unknown stochastic process.



### Step 1: Recognize the SDE Type

This is a **geometric Brownian motion (GBM)**, commonly used in financial modeling (e.g., the Black-Scholes model).

It is a **linear SDE**, and we can solve it using **Itô's Lemma** (or method of integrating factors).



### Step 2: Solve via Itô's Lemma

Let us define a new process:

$$
Y_t = \ln X_t.
$$

Apply **Itô’s Lemma** to $Y_t = \ln X_t$:

$$
dY_t = \frac{1}{X_t} dX_t - \frac{1}{2} \frac{1}{X_t^2} (dX_t)^2.
$$

From the SDE:

$$
dX_t = \mu X_t  dt + \sigma X_t  dW_t,
$$

so,

$$
(dX_t)^2 = (\sigma X_t)^2 (dW_t)^2 = \sigma^2 X_t^2  dt,
$$

since $(dW_t)^2 = dt$.

Substitute into Itô’s Lemma:

$$
dY_t = \frac{1}{X_t}(\mu X_t  dt + \sigma X_t  dW_t) - \frac{1}{2} \frac{1}{X_t^2} (\sigma^2 X_t^2  dt),
$$

$$
dY_t = \mu  dt + \sigma  dW_t - \frac{1}{2} \sigma^2  dt,
$$

$$
\boxed{dY_t = \left(\mu - \frac{1}{2} \sigma^2 \right) dt + \sigma  dW_t}.
$$



### Step 3: Integrate the SDE for $Y_t$

This is now a simple linear SDE:

$$
Y_t = Y_0 + \left(\mu - \frac{1}{2} \sigma^2 \right)t + \sigma W_t,
$$

where $Y_0 = \ln X_0$. Therefore:

$$
Y_t = \ln X_0 + \left(\mu - \frac{1}{2} \sigma^2 \right)t + \sigma W_t.
$$

Exponentiate both sides to recover $X_t$:

$$
X_t = \exp(Y_t) = \exp\left(\ln X_0 + \left(\mu - \frac{1}{2} \sigma^2 \right)t + \sigma W_t\right),
$$

$$
\boxed{X_t = X_0 \exp\left( \left(\mu - \frac{1}{2} \sigma^2 \right)t + \sigma W_t \right)}.
$$



### ✅ Final Solution

$$
\boxed{X_t = X_0 \exp\left( \left(\mu - \frac{1}{2} \sigma^2 \right)t + \sigma W_t \right)}
$$

This is the unique strong solution of the SDE $dX_t = \mu X_t  dt + \sigma X_t  dW_t$.


---

## How do you simulate an SDE numerically? Describe Euler-Maruyama and Milstein methods.

To simulate a stochastic differential equation (SDE) numerically, we discretize the continuous-time process using methods like **Euler–Maruyama** or **Milstein**. These methods approximate the solution by stepping through time in small increments and estimating the stochastic process’s evolution.



## 🎯 General SDE Form

$$
dX_t = a(X_t, t)  dt + b(X_t, t)  dW_t,
$$

where:

* $a(X_t, t)$: **drift** term,
* $b(X_t, t)$: **diffusion** term,
* $W_t$: standard Brownian motion,
* $X_0 = x_0$: initial condition.

We want to approximate $X_t$ at discrete times $t_0, t_1, ..., t_N$ with step size $\Delta t = t_{n+1} - t_n$.



## 1. **Euler–Maruyama Method**

This is the simplest and most widely used numerical method for SDEs. It's analogous to the Euler method for ODEs, but adapted for stochastic calculus.

### **Update formula**:

$$
X_{n+1} = X_n + a(X_n, t_n) \Delta t + b(X_n, t_n) \Delta W_n,
$$

where:

* $\Delta W_n = W_{t_{n+1}} - W_{t_n} \sim \mathcal{N}(0, \Delta t)$: a normal random variable with mean 0 and variance $\Delta t$.

### **Use case**:

* Fast and easy to implement.
* Converges weakly with order 1 and strongly with order 0.5.
* Good for qualitative understanding but less accurate for precise simulations.



## 2. **Milstein Method**

The Milstein method improves upon Euler–Maruyama by accounting for the **Itô correction** term, which arises when the diffusion term $b(X_t)$ depends on $X_t$.

### **Update formula**:

$$
X_{n+1} = X_n + a(X_n, t_n) \Delta t + b(X_n, t_n) \Delta W_n + \frac{1}{2} b(X_n, t_n) b'(X_n, t_n) \left((\Delta W_n)^2 - \Delta t\right),
$$

where:

* $b'(X_n, t_n)$ is the derivative of $b$ with respect to $X$.

### **Use case**:

* Stronger convergence (order 1.0).
* More accurate than Euler–Maruyama when the diffusion term is nonlinear.
* Requires computing $b'$, which might be complex for some models.



## 🔁 Simulation Steps (for both methods)

1. Set initial condition $X_0 = x_0$, time interval $[0, T]$, and step size $\Delta t = T/N$.
2. Generate standard normal random variables $\Delta W_n \sim \mathcal{N}(0, \Delta t)$.
3. Iterate the chosen method (Euler–Maruyama or Milstein) for $n = 0, 1, ..., N-1$.
4. Record $X_n$ at each step.



## 🔍 Example: Simulating Geometric Brownian Motion

$$
dX_t = \mu X_t dt + \sigma X_t dW_t.
$$

* **Euler–Maruyama**:

$$
  X_{n+1} = X_n + \mu X_n \Delta t + \sigma X_n \Delta W_n.
$$

* **Milstein**:

$$
  X_{n+1} = X_n + \mu X_n \Delta t + \sigma X_n \Delta W_n + \frac{1}{2} \sigma^2 X_n \left((\Delta W_n)^2 - \Delta t\right).
$$

In this case, $b(X) = \sigma X \Rightarrow b'(X) = \sigma$.

---

## 🧠 Summary Table

| Method         | Strong Convergence | Needs Derivative of $b$? | Accuracy      | Use Case                 |
| -------------- | ------------------ | ------------------------ | ------------- | ------------------------ |
| Euler–Maruyama | 0.5                | ❌                        | Basic         | Simple models, fast runs |
| Milstein       | 1.0                | ✅                        | More accurate | Nonlinear diffusion      |


Python Example:


### ✅ Observations
- Milstein generally tracks the exact solution more accurately than Euler, especially when the time step is not tiny.

- Differences become more noticeable in high-volatility or long-time scenarios.

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
mu = 0.1
sigma = 0.2
X0 = 1.0
T = 1.0
N = 1000
dt = T / N
t = np.linspace(0, T, N+1)
M = 5  # Number of paths

# Prepare arrays
X_euler = np.zeros((M, N+1))
X_milstein = np.zeros((M, N+1))
X_exact = np.zeros((M, N+1))

X_euler[:, 0] = X0
X_milstein[:, 0] = X0
X_exact[:, 0] = X0

# Simulate paths
for m in range(M):
    W = np.zeros(N+1)
    for i in range(1, N+1):
        dW = np.random.normal(0, np.sqrt(dt))
        W[i] = W[i-1] + dW

        # Euler–Maruyama
        X_euler[m, i] = X_euler[m, i-1] + mu * X_euler[m, i-1] * dt + sigma * X_euler[m, i-1] * dW

        # Milstein
        X_prev = X_milstein[m, i-1]
        X_milstein[m, i] = X_prev + mu * X_prev * dt + sigma * X_prev * dW + 0.5 * sigma**2 * X_prev * ((dW)**2 - dt)

        # Exact
        X_exact[m, i] = X0 * np.exp((mu - 0.5 * sigma**2) * t[i] + sigma * W[i])

# Plot results
plt.figure(figsize=(12, 6))
for m in range(M):
    plt.plot(t, X_exact[m], 'k--', label='Exact' if m == 0 else "", alpha=0.9)
    plt.plot(t, X_euler[m], 'b-', label='Euler' if m == 0 else "", alpha=0.7)
    plt.plot(t, X_milstein[m], 'g-.', label='Milstein' if m == 0 else "", alpha=0.7)

plt.title("Geometric Brownian Motion: Exact vs Euler vs Milstein")
plt.xlabel("Time")
plt.ylabel("X(t)")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
```
---


## Describe the Fokker-Planck equation and its relation to SDEs.

The **Fokker–Planck equation** (also known as the **forward Kolmogorov equation**) describes the **time evolution of the probability density function (PDF)** of a stochastic process governed by a **stochastic differential equation (SDE)**.



### 🎯 General SDE

Consider the Itô SDE:

$$
dX_t = a(X_t, t)  dt + b(X_t, t)  dW_t,
$$

with:

* Drift term $a(X_t, t)$,
* Diffusion term $b(X_t, t)$,
* Brownian motion $W_t$.

Let $p(x, t)$ be the **probability density function** of $X_t$, i.e., $p(x, t)   dx = \mathbb{P}(x \leq X_t \leq x + dx)$.



### 🧮 Fokker–Planck Equation (Forward Kolmogorov)

$$
\frac{\partial p(x, t)}{\partial t}
= - \frac{\partial}{\partial x} \left[ a(x, t)  p(x, t) \right]
+ \frac{1}{2} \frac{\partial^2}{\partial x^2} \left[ b^2(x, t)  p(x, t) \right]
$$

---

### 🔁 Intuition

* **Drift term $a(x, t)$** → transports probability mass (like advection in fluid flow).
* **Diffusion term $b^2(x, t)$** → spreads out the distribution (like diffusion in physics).

This partial differential equation (PDE) tells how the **distribution of the solution** to the SDE evolves over time.

---

### 📌 Special Case: Constant Coefficients

For the SDE:

$$
dX_t = \mu  dt + \sigma  dW_t,
$$

with constant drift $\mu$ and diffusion $\sigma$, the Fokker–Planck equation becomes:

$$
\frac{\partial p}{\partial t}
= - \mu \frac{\partial p}{\partial x}
+ \frac{1}{2} \sigma^2 \frac{\partial^2 p}{\partial x^2}.
$$

This is a **linear convection–diffusion equation**, and its solution is a Gaussian PDF with:

* Mean $\mu t$,
* Variance $\sigma^2 t$.

---

### 🧠 Relation to SDEs

* The **SDE** describes how **sample paths** (individual realizations) evolve.
* The **Fokker–Planck equation** describes how the **distribution** of those paths evolves.

They are two complementary views of the same stochastic process:

* **SDE → pathwise (microscopic) behavior**.
* **Fokker–Planck → distributional (macroscopic) behavior**.

---

### 🧪 Analogy

Think of particles in a fluid:

* The SDE tracks **one particle’s random movement**.
* The Fokker–Planck equation describes **how the whole cloud of particles spreads over time**.

---






## What is a martingale? Show that Brownian motion is a martingale.

A **martingale** is a fundamental concept in probability theory and stochastic processes, especially in the study of **stochastic differential equations** and **financial mathematics**.

---

## 🎯 Definition: Martingale

A stochastic process $(X_t)_{t \geq 0}$ adapted to a filtration $(\mathcal{F}_t)_{t \geq 0}$ is called a **martingale** if:

1. $X_t$ is integrable: $\mathbb{E}[|X_t|] < \infty$ for all $t$,
2. $X_t$ is adapted (you only use past and present information),
3. For all $0 \leq s < t$,

$$
   \boxed{\mathbb{E}[X_t \mid \mathcal{F}_s] = X_s}
$$

   — the **conditional expectation of the future value equals the current value**.

> 🔁 Intuition: A martingale is a "fair game" — there's no expected gain or loss over time, given the present.

---

## ✅ Brownian Motion as a Martingale

Let $(W_t)_{t \geq 0}$ be a **standard Brownian motion**, i.e.:

* $W_0 = 0$,
* $W_t$ has independent increments,
* $W_t - W_s \sim \mathcal{N}(0, t - s)$,
* $W_t$ is continuous and adapted.

### We want to show:

$$
\mathbb{E}[W_t \mid \mathcal{F}_s] = W_s \quad \text{for } 0 \leq s < t.
$$

---

## 🧠 Proof Sketch

Let $s < t$. Since Brownian increments are independent of the past:

$$
W_t = W_s + (W_t - W_s)
$$

And because $W_t - W_s \sim \mathcal{N}(0, t - s)$ is independent of $\mathcal{F}_s$, we have:

$$
\mathbb{E}[W_t \mid \mathcal{F}_s]
= \mathbb{E}[W_s + (W_t - W_s) \mid \mathcal{F}_s]
= W_s + \mathbb{E}[W_t - W_s \mid \mathcal{F}_s]
= W_s + 0 = W_s
$$

✅ Therefore, $W_t$ is a martingale.

---

## 📎 Related Facts

| Process                                   | Martingale? | Why?                                         |
| ----------------------------------------- | ----------- | -------------------------------------------- |
| $W_t$                                     | ✅ Yes       | Zero mean increment, independent increments  |
| $W_t^2 - t$                               | ✅ Yes       | Itô calculus shows it has zero drift         |
| $e^{\mu t + \sigma W_t}$                  | ❌ No        | Not a martingale (grows exponentially)       |
| $e^{-\frac{1}{2}\sigma^2 t + \sigma W_t}$ | ✅ Yes       | Martingale due to Girsanov/Novikov condition |

---


## verify whether a process is a martingale using Itô’s Lemma

## 🔁 Recap: Itô's Lemma (1D Version)

Let $X_t$ be an Itô process:

$$
dX_t = a(t, X_t) dt + b(t, X_t) dW_t,
$$

and let $f(t, X_t)$ be a smooth function. Then:

$$
df(t, X_t) = \left( \frac{\partial f}{\partial t} + a \frac{\partial f}{\partial x} + \frac{1}{2} b^2 \frac{\partial^2 f}{\partial x^2} \right) dt
+ b \frac{\partial f}{\partial x} dW_t.
$$

---

## ✅ Martingale Criterion via Itô’s Lemma

A process $Y_t = f(t, X_t)$ is a **martingale** if:

* It is integrable,
* Its **drift term (the dt term in Itô’s lemma) is zero**.

---

## 📌 Example: Show that $M_t = W_t^2 - t$ is a martingale

### Step 1: Define the process

Let $X_t = W_t$, so $dX_t = dW_t$. Take:

$$
f(X_t) = W_t^2
\Rightarrow M_t = f(W_t) - t.
$$

We want to check whether $M_t$ is a martingale.

---

### Step 2: Apply Itô’s Lemma

Let $f(x) = x^2$, and $X_t = W_t$, so:

$$
df(W_t) = f'(W_t) dW_t + \frac{1}{2} f''(W_t) (dW_t)^2
= 2W_t dW_t + \frac{1}{2}(2) dt = 2W_t dW_t + dt.
$$

So:

$$
d(W_t^2) = 2W_t dW_t + dt
\Rightarrow d(W_t^2 - t) = 2W_t dW_t.
$$

### Step 3: Check the drift term

The $dt$ term is **gone** in the differential of $W_t^2 - t$, so:

$$
dM_t = 2W_t dW_t \quad \Rightarrow \text{No drift!}
$$

✅ Therefore, $M_t = W_t^2 - t$ is a **martingale**.

---

## ⚠️ Example of a Non-Martingale

Try $Y_t = W_t^2$. Its Itô differential is:

$$
d(W_t^2) = 2W_t dW_t + dt.
$$

The drift term $dt$ is nonzero → **not a martingale**.

---

## 🔍 Summary: How to Check a Martingale with Itô's Lemma

1. **Write** the process as $f(t, X_t)$,
2. **Apply Itô’s lemma**,
3. If the **dt (drift) term is zero**, it’s a **local martingale**,
4. If it’s also integrable → it’s a **martingale**.




---

## 📐 Definition: σ-Algebra

A **σ-algebra** $\mathcal{F}$ over a sample space $\Omega$ is a collection of subsets of $\Omega$ (i.e., events) satisfying:

1. **The full set is included**:

$$
   \Omega \in \mathcal{F}
$$

2. **Closed under complementation**:
   If $A \in \mathcal{F}$, then $\Omega \setminus A \in \mathcal{F}$.

3. **Closed under countable unions**:
   If $A_1, A_2, A_3, \dots \in \mathcal{F}$, then:

$$
   \bigcup_{n=1}^\infty A_n \in \mathcal{F}
$$

> ➕ From these, it follows that it's also closed under **countable intersections** and **set differences**.

---

## 🔍 Intuition

* A σ-algebra represents a **set of events** for which we can assign probabilities.
* Think of it as the mathematical formalization of "what we know" or "what we can measure".
* In probability, the **probability space** is $(\Omega, \mathcal{F}, \mathbb{P})$, where:

  * $\Omega$: sample space
  * $\mathcal{F}$: σ-algebra of events
  * $\mathbb{P}$: probability measure

---

## 📦 Example

Let’s take a simple sample space:

$$
\Omega = \{H, T\} \quad \text{(coin flip)}
$$

The **smallest** σ-algebra is:

$$
\mathcal{F}_1 = \{ \emptyset, \Omega \}
$$

(you know nothing except whether the coin flipped at all)

A **larger** σ-algebra could be:

$$
\mathcal{F}_2 = \{ \emptyset, \{H\}, \{T\}, \Omega \}
$$

(you know exactly which side came up)

---

## 📈 In Stochastic Processes

In processes like Brownian motion:

* The σ-algebra $\mathcal{F}_t = \sigma(W_s : 0 \leq s \leq t)$ represents **everything observable** up to time $t$.
* It's used to define:

  * **Adapted processes**
  * **Stopping times**
  * **Martingales**
  * **Conditional expectations**

---

## 📌 Summary

| Concept                 | Description                                                                       |
| ----------------------- | --------------------------------------------------------------------------------- |
| σ-algebra               | A structured collection of sets closed under complementation and countable unions |
| Purpose                 | Defines what events are "measurable" or "knowable"                                |
| In probability          | Needed to assign probabilities and define random variables                        |
| In stochastic processes | Encodes available information over time                                           |



---


Certainly! Filtration and stopping times are foundational concepts in **stochastic processes**, especially in martingale theory and stochastic calculus.

---
## Define filtration and stopping times.

## 📂 1. **Filtration**

### 🔍 Definition:

A **filtration** $(\mathcal{F}_t)_{t \geq 0}$ is a family of **increasing σ-algebras** representing the accumulation of information over time.

$$
\mathcal{F}_s \subseteq \mathcal{F}_t \quad \text{for all } 0 \leq s \leq t
$$

Each $\mathcal{F}_t$ contains all information **observable** up to time $t$.

---

### 💡 Intuition:

Think of $\mathcal{F}_t$ as the "information available to an observer" at time $t$.
If you're watching a Brownian motion evolve, you see more and more of its path as $t$ increases.

---

### 📌 Example:

Let $W_t$ be a Brownian motion. The **natural filtration** of $W$ is:

$$
\mathcal{F}_t^W = \sigma(W_s : 0 \leq s \leq t),
$$

which is the smallest σ-algebra making all $W_s$, $s \leq t$, measurable.

---

## ⏱️ 2. **Stopping Time**

### 🔍 Definition:

A **stopping time** $\tau$ (with respect to a filtration $(\mathcal{F}_t)$) is a random time such that:

$$
\{\tau \leq t\} \in \mathcal{F}_t \quad \text{for all } t \geq 0.
$$

That is, **at any time $t$**, you can tell whether the event "$\tau \leq t$" has occurred **using only the information up to time $t$**.

---

### 💡 Intuition:

A stopping time is a "decision time" based only on **current and past** information — you can’t peek into the future.

---

### 📌 Examples:

| Stopping Time                              | Description                                                 |
| ------------------------------------------ | ----------------------------------------------------------- |
| $\tau = \inf\{t \geq 0 : W_t \geq 1\}$     | First time Brownian motion hits level 1 — ✅ a stopping time |
| $\tau = \inf\{t \geq 0 : W_{t+1} \geq 1\}$ | Depends on future info — ❌ not a stopping time              |
| $\tau = T$, where $T$ is deterministic     | Always a stopping time                                      |

---

### 🧠 Stopping Times Are Used For:

* Defining optional stopping theorems,
* Localizing martingales (e.g., stopping them to get a true martingale),
* Modifying stochastic processes (e.g., $W_{t \wedge \tau}$).

---

## 🔗 Connection: Filtration & Stopping Times

* A stopping time is **defined with respect to a filtration**.
* If you change the filtration, you may change whether something is a stopping time.
* Stopping times let you “pause” a process when a certain **data-dependent** condition occurs.

---

## Explain Girsanov’s Theorem and its application in risk-neutral pricing.


**Girsanov’s Theorem** is a cornerstone result in stochastic calculus, particularly useful in **mathematical finance**. It provides a way to **change the probability measure** so that a process with drift becomes a **martingale** under the new measure — this is key to **risk-neutral pricing**.

---

## 🎯 What Is Girsanov’s Theorem?

### 🔍 **Informal Statement:**

Let $(W_t)_{t \geq 0}$ be a **Brownian motion** under a probability measure $\mathbb{P}$.
Suppose you define a new process:

$$
\tilde{W}_t = W_t - \int_0^t \theta_s   ds
$$

Then under a **new measure** $\mathbb{Q}$, defined via a Radon-Nikodym derivative, $\tilde{W}_t$ becomes a **Brownian motion**.

This is the core of **Girsanov’s Theorem**: you can change the drift of a Brownian motion by switching to an equivalent probability measure.

---

## 🧾 Formal Version (Simplified Case)

Let $W_t$ be Brownian motion under $\mathbb{P}$, and let $\theta(t)$ be an adapted process satisfying Novikov's condition:

$$
\mathbb{E}^\mathbb{P} \left[ \exp\left( \frac{1}{2} \int_0^T \theta_s^2 ds \right) \right] < \infty.
$$

Then define the **likelihood ratio (density)**:

$$
\frac{d\mathbb{Q}}{d\mathbb{P}} = Z_T = \exp\left( -\int_0^T \theta_s  dW_s - \frac{1}{2} \int_0^T \theta_s^2 ds \right).
$$

Under the new measure $\mathbb{Q}$, the process

$$
\tilde{W}_t := W_t + \int_0^t \theta_s  ds
$$

is a **Brownian motion** under $\mathbb{Q}$.

---

## 🏦 Application: Risk-Neutral Pricing

In finance, asset prices often follow a **geometric Brownian motion** under the **real-world measure** $\mathbb{P}$:

$$
dS_t = \mu S_t dt + \sigma S_t dW_t^\mathbb{P}
$$

But for **pricing derivatives**, we want to work under a **risk-neutral measure** $\mathbb{Q}$, where the asset grows at the **risk-free rate $r$** instead of the real-world drift $\mu$:

$$
dS_t = r S_t dt + \sigma S_t dW_t^\mathbb{Q}
$$

### 🚀 How Girsanov Helps:

We use **Girsanov's theorem** to change the drift from $\mu$ to $r$. That is, we define:

$$
\theta = \frac{\mu - r}{\sigma}
$$

Then define the new measure $\mathbb{Q}$ by:

$$
\frac{d\mathbb{Q}}{d\mathbb{P}} = \exp\left( -\theta W_T^\mathbb{P} - \frac{1}{2} \theta^2 T \right)
$$

Under $\mathbb{Q}$, the process:

$$
W_t^\mathbb{Q} := W_t^\mathbb{P} + \theta t
$$

is a Brownian motion, and the SDE becomes:

$$
dS_t = r S_t dt + \sigma S_t dW_t^\mathbb{Q}
$$

✅ This allows us to use the **risk-neutral pricing formula**:

$$
\boxed{ \text{Price}_t = \mathbb{E}^{\mathbb{Q}} \left[ e^{-r(T - t)} \cdot \text{Payoff}(S_T) \mid \mathcal{F}_t \right] }
$$

---

## 📌 Summary

| Concept                      | Description                                                                                 |
| ---------------------------- | ------------------------------------------------------------------------------------------- |
| **Girsanov's Theorem**       | Allows us to change the measure to eliminate drift in Brownian motion                       |
| **New Measure $\mathbb{Q}$** | Equivalent to $\mathbb{P}$, but drift changes                                               |
| **Application**              | Risk-neutral pricing: price derivatives as expected values under $\mathbb{Q}$               |
| **Result**                   | Asset grows at risk-free rate under $\mathbb{Q}$, making discounted prices into martingales |

---

