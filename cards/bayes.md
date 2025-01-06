### Bayesian Inference

---

**Bayesian inference** is a statistical method based on **Bayes' Theorem**, which provides a way to update the probability estimate for a hypothesis as more evidence or data becomes available. It is a cornerstone of probabilistic reasoning and is widely used in machine learning, data science, and statistics.

#### **Bayes' Theorem Formula**

$$P(H|E) = \frac{P(E|H) \cdot P(H)}{P(E)},$$


where:

- $P(H|E)$: **Posterior probability**. The probability of the hypothesis $H$ given the evidence $E$.
- $P(H)$: **Prior probability**. The initial probability of the hypothesis before observing any evidence.
- $P(E|H)$: **Likelihood**. The probability of observing the evidence $E$ given the hypothesis $H$ is true.
- $P(E)$: **Marginal likelihood** or evidence. The total probability of observing the evidence under all possible hypotheses.

#### **How It Works**
Bayesian inference adjusts the **prior belief** $P(H)$ based on new evidence $E$ to compute the **posterior belief** $P(H|E)$. It allows us to make predictions or decisions in the presence of uncertainty.

