explain each and every values in detail in the following statsmodel regression output

```
                           OLS Regression Results                            
==============================================================================
Dep. Variable:                      y   R-squared:                       0.362
Model:                            OLS   Adj. R-squared:                  0.357
Method:                 Least Squares   F-statistic:                     78.83
Date:                Mon, 02 Jun 2025   Prob (F-statistic):           2.07e-40
Time:                        15:11:45   Log-Likelihood:                 689.31
No. Observations:                 421   AIC:                            -1371.
Df Residuals:                     417   BIC:                            -1354.
Df Model:                           3                                         
Covariance Type:            nonrobust                                         
==============================================================================
                 coef    std err          t      P>|t|      [0.025      0.975]
------------------------------------------------------------------------------
const          0.0059      0.002      2.521      0.012       0.001       0.010
MKT-RF         0.7599      0.054     14.030      0.000       0.653       0.866
SMB           -0.4919      0.078     -6.321      0.000      -0.645      -0.339
HML            0.4041      0.075      5.409      0.000       0.257       0.551
==============================================================================
Omnibus:                       90.882   Durbin-Watson:                   2.115
Prob(Omnibus):                  0.000   Jarque-Bera (JB):              228.408
Skew:                           1.060   Prob(JB):                     2.52e-50
Kurtosis:                       5.920   Cond. No.                         36.5
==============================================================================


```


---

### **Top-Level Information:**

#### **Dep. Variable: y**

* This is the dependent variable (response variable) you're trying to predict or explain — denoted as `y` here.

#### **Model: OLS**

* Ordinary Least Squares regression is the method used to estimate the parameters.

#### **Method: Least Squares**

* Specifies the estimation method again — it minimizes the sum of squared residuals.

#### **Date & Time**

* The timestamp when the regression was executed.

---

### **Model Fit Statistics:**

#### **R-squared: 0.362**

* **R² (Coefficient of Determination)** indicates that 36.2% of the variability in the dependent variable (`y`) can be explained by the independent variables (`MKT-RF`, `SMB`, `HML`).
* Values range from 0 to 1, with higher values indicating a better fit.

#### **Adj. R-squared: 0.357**

* Adjusted R² adjusts for the number of predictors in the model and penalizes overfitting.
* It’s useful when comparing models with different numbers of variables.

#### **F-statistic: 78.83**

* Tests the **overall significance** of the model — whether at least one predictor variable has a non-zero coefficient.
* A high F-statistic indicates that the model is a good fit.

#### **Prob (F-statistic): 2.07e-40**

* The **p-value** associated with the F-statistic.
* This extremely small value indicates the model is statistically significant (rejects the null hypothesis that all coefficients are zero).

---

### **Log-Likelihood: 689.31**

* A measure of model fit. It’s used to compute AIC and BIC. Higher values suggest a better model.

#### **AIC: -1371.0 (Akaike Information Criterion)**

#### **BIC: -1354.0 (Bayesian Information Criterion)**

* Lower AIC/BIC values indicate better model fit.
* These criteria penalize model complexity to avoid overfitting.

---

### **Model Degrees of Freedom:**

#### **No. Observations: 421**

* The number of observations (data points) used in the regression.

#### **Df Model: 3**

* The number of **independent variables** (not including the intercept): `MKT-RF`, `SMB`, `HML`.

#### **Df Residuals: 417**

* Degrees of freedom of residuals = `No. Observations - Df Model - 1`.

---

### **Coefficient Table:**

```
\| Term     | coef  | std err | t      | P>|t| | \[0.025, 0.975]        |
\|----------|--------|----------|--------|--------|-------------------------|
\| `const`  | 0.0059 | 0.002    | 2.521  | 0.012  | (0.001, 0.010)          |
\| `MKT-RF` | 0.7599 | 0.054    | 14.030 | 0.000  | (0.653, 0.866)          |
\| `SMB`    | -0.4919| 0.078    | -6.321 | 0.000  | (-0.645, -0.339)        |
\| `HML`    | 0.4041 | 0.075    | 5.409  | 0.000  | (0.257, 0.551)          |

```


#### Each column means:

* **coef**: Estimated regression coefficient.

  * E.g., Holding others constant, a one-unit increase in `MKT-RF` increases `y` by \~0.76 units.
* **std err**: Standard error of the coefficient — indicates uncertainty in the estimate.
* **t**: t-statistic = `coef / std err`, used to test if the coefficient is significantly different from 0.
* **P>|t|**: p-value — if < 0.05, the coefficient is statistically significant.
* **\[0.025, 0.975]**: 95% confidence interval for the coefficient.

---

### **Diagnostic Tests:**

#### **Omnibus: 90.882**

* A combined test for skewness and kurtosis of residuals. Large values indicate non-normality.

#### **Prob(Omnibus): 0.000**

* p-value for the Omnibus test. Very small — residuals are **not normally distributed**.

#### **Jarque-Bera (JB): 228.408**

* Another test for normality of residuals. Combines skewness and kurtosis.

#### **Prob(JB): 2.52e-50**

* The very low p-value means residuals **deviate significantly from normality**.

#### **Skew: 1.060**

* Measures asymmetry of the residuals. A value > 0 indicates a **right-skewed distribution**.

#### **Kurtosis: 5.920**

* Measures the "tailedness" of the distribution. Normal distribution has kurtosis = 3.
* Value > 3 suggests **heavy tails** (more outliers).

#### **Durbin-Watson: 2.115**

* Tests for autocorrelation in residuals.

  * \~2 indicates no autocorrelation (good).
  * <2 suggests positive autocorrelation; >2 suggests negative.

#### **Cond. No.: 36.5**

* Condition Number — measures multicollinearity.

  * Values > 30 might indicate **moderate** multicollinearity.
  * Very high values (>100) signal severe multicollinearity issues.

---

Let me know if you’d like a visual explanation or how to interpret this in finance context (e.g., Fama-French model).
