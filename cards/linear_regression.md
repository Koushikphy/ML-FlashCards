### What is Linear Regression

---

Linear regression is a statistical method used to model the relationship between a dependent variable (target) and one or more independent variables (predictors) by fitting a linear equation to observed data. The goal is to find the best-fitting line (or hyperplane) that minimizes the difference between the predicted and actual values. It is commonly used for prediction, trend analysis, and forecasting. 

$$y=\beta_0+\beta X + \epsilon$$    
$$\beta = (X^T X)^{-1}X^T y$$

- $y$ is the target variable
- $X$ is the matrix of predictor variables
- $\beta$ is the coefficient vector
- $\beta_0$ is the intercept
- $\epsilon$ represent the error.



#### sample python calculation

x = np.linspace(0,1)  
y = x**2 + .5*x + np.random.rand(len(x))/25  

X = np.matrix([x**2, x, np.ones_like(x) ]).T   
Y = np.matrix(y).T

np.linalg.inv(X.T*X)*X.T*Y  
