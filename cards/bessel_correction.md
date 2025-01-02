### Bessel's Correction

---


Bessel's Correction is a technique used in statistics to correct the bias in the estimation of the population variance and standard deviation from a sample. It is particularly important when working with small sample sizes.  It involves using n-1 instead of n as the denominator in the formula for sample variance, where n is the sample size.

### Why Bessel's Correction is Needed:
When you calculate the variance or standard deviation of a sample, you are trying to estimate the variance or standard deviation of the entire population from which the sample was drawn. If you were to use the sample mean to calculate the variance, you would tend to **underestimate** the population variance, especially when the sample size is small. This happens because the sample mean is typically closer to the sample data points than the true population mean, which leads to smaller squared deviations.

### Formula for Variance with and without Bessel's Correction:
1. **Without Bessel's Correction** (biased estimator):
   The formula for the **sample variance** $s^2$ without Bessel's correction is:
   $$s^2 = \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2$$
   where $x_i$ = individual data points, $\bar{x}$ = sample mean, $n$ = sample size.  
   This formula uses $n$ in the denominator, which tends to underestimate the population variance.

2. **With Bessel's Correction** (unbiased estimator):
   To correct this bias, Bessel’s correction uses $n - 1$ (degrees of freedom) instead of $n$. The corrected formula for the **sample variance** is:
   $$s^2 = \frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2$$
   By dividing by $n - 1$, this correction makes the sample variance an **unbiased estimator** of the population variance. This ensures that, on average, the sample variance is equal to the true population variance when applied to multiple samples.

### Explanation:
- **Degrees of Freedom**: The term $n - 1$ represents the number of independent pieces of information available to estimate the population variance. The sample mean $\bar{x}$ is calculated from the data, so it is already constrained by the sample. This reduces the number of independent deviations by 1, hence $n - 1$ degrees of freedom.
