
### Can a Random Forest ever outperform XGBoost? If yes, when?

Yes, **Random Forest can outperform XGBoost** in certain scenarios:

* **Small or clean datasets**: If the dataset is small, relatively noise-free, and doesn't require complex modeling, Random Forest may perform comparably or better due to its simplicity and lower risk of overfitting.
* **Less tuning required**: Random Forest often works well with default parameters, whereas XGBoost usually requires more careful hyperparameter tuning.
* **Noisy data**: In cases with high noise, Random Forest may generalize better because it's less aggressive than boosting methods, which can overfit noisy signals.
* **High latency sensitivity**: In real-time applications where inference time is critical, Random Forests can be faster depending on the model size.

However, in most structured data problems, XGBoost tends to outperform Random Forest due to its boosting mechanism and built-in handling of missing values and regularization.

---

### What does the coefficient in logistic regression mean?

In **logistic regression**, the coefficient represents the **log-odds change** of the outcome for a one-unit increase in the predictor, holding all other variables constant.

Mathematically:

* If $\beta_i$ is the coefficient of feature $x_i$, then a one-unit increase in $x_i$ multiplies the **odds** of the outcome by $e^{\beta_i}$.
* Positive $\beta_i$: increases the odds (more likely the positive class).
* Negative $\beta_i$: decreases the odds (more likely the negative class).

Example:

* If $\beta_i = 0.7$, then each unit increase in $x_i$ multiplies the odds by $e^{0.7} \approx 2.01$, effectively doubling the odds.

---

### When should you do train-test split: before or after feature engineering?

Train-test split should be done *before* feature engineering.

**Why:**

* To **prevent data leakage**: If feature engineering is done on the entire dataset, you risk incorporating information from the test set into the features, leading to overly optimistic performance estimates.
* Proper procedure:

  1. Split data into train and test.
  2. Fit preprocessing steps (e.g., imputation, scaling, encoding) **only on the training set**.
  3. Apply the same fitted transformations to the test set.

**Exception**: In cross-validation pipelines, tools like `sklearn`'s `Pipeline` ensure transformations are applied correctly, even if the split seems to come later in code.

---

Excellent set of in-depth questions. Here's how I would answer each of them in a professional, interview-style format.

---

### Why might adding more features to a dataset degrade model performance? How would you identify and remove irrelevant features?

Adding more features can:

* Introduce **noise**, especially if features are irrelevant or redundant.
* Increase the **dimensionality**, leading to the **curse of dimensionality**, making it harder for the model to generalize.
* Cause **overfitting**, especially in small datasets.

To identify/remove irrelevant features:

* Use **feature importance** scores (e.g., from tree models).
* Apply **L1 regularization** (Lasso) to shrink irrelevant coefficients to zero.
* Perform **Recursive Feature Elimination (RFE)**.
* Use **correlation analysis** or **mutual information**.

---

### Why is gradient descent not guaranteed to find the global minimum in non-convex loss functions?

Non-convex functions have:

* **Multiple local minima and saddle points**.
* Gradient descent is a local optimization method—it follows the negative gradient and can get stuck in **local minima or plateaus** depending on initialization.
* No guarantee of reaching the **global** minimum unless the function is convex.

---

### Why might you prefer a simpler model like Logistic Regression over a complex one like a Neural Network in some cases?

* **Interpretability**: Coefficients in logistic regression are easy to interpret.
* **Faster training/inference**: Especially on small datasets or with limited compute.
* **Lower risk of overfitting** on small or linearly separable datasets.
* **Baseline performance**: Logistic regression often serves as a strong, fast baseline.

---

### If your model's AUC score is high but precision is low, why might this happen, and how would you address it?

* **AUC measures ranking ability**, not absolute performance at a specific threshold.
* Low **precision** suggests many false positives, possibly due to:

  * Class imbalance.
  * Poor threshold choice.

**Solutions:**

* Adjust the **decision threshold** based on precision-recall tradeoff.
* Use **Precision-Recall curves** rather than ROC.
* Consider **sampling techniques** or **cost-sensitive learning**.

---

### 5. Why do tree-based algorithms like XGBoost handle missing values better than most other models?

* XGBoost automatically learns the **best direction (left/right)** to send missing values during tree splits.
* No need for explicit imputation.
* This allows it to **preserve patterns** in missingness that may carry predictive value.


---

###  is it important to check the distribution of residuals in regression analysi

* Residuals should be **random and normally distributed** for assumptions of linear regression to hold.
* Non-normality or patterns indicate:

  * **Model misspecification**
  * **Heteroscedasticity**
  * **Omitted variables**

---

### If two datasets have the same mean and variance, why might they still have very different distributions?

* They can differ in:

  * **Skewness** (asymmetry)
  * **Kurtosis** (tailedness)
  * **Multimodality**
* Summary statistics are not sufficient—**visualization (e.g., histograms, boxplots)** helps capture distributional shape.

---

### Why is it crucial to consider sample size when interpreting confidence intervals?

* Smaller sample sizes lead to **wider confidence intervals**, reflecting greater uncertainty.
* Confidence intervals shrink with more data, increasing **estimation precision**.
* Misinterpretation can lead to **overconfidence in results**.

---

### In A/B testing, why might a test that runs for too long lead to misleading results?

* Increases risk of **peeking** or **p-hacking** (false positives).
* External factors (seasonality, user behavior drift) can contaminate results.
* Recommended to predefine **stopping rules** and use **sequential testing** methods if needed.

---

### Why might you use bootstrapping instead of traditional hypothesis testing for small datasets?

* Bootstrapping is **non-parametric** and doesn’t assume normality.
* More reliable when parametric assumptions don’t hold.
* Allows estimating **confidence intervals** and **standard errors** from limited data.

---

### Your model is biased against certain demographics. Why might this happen, and how would you mitigate it?

* Causes:

  * **Imbalanced data** or historical bias.
  * Proxy variables encoding sensitive attributes.
  * Objective function optimizing for overall accuracy.

**Mitigation:**

* Use **fairness-aware algorithms**.
* Monitor metrics like **equal opportunity** or **demographic parity**.
* Reweigh samples or apply **adversarial debiasing**.

---

### You’re tasked with building a recommendation system for a new e-commerce site with no historical data. How would you approach this?

* Use **cold start strategies**:

  * **Content-based filtering** using item/user metadata.
  * **Popular/trending items** as default.
  * Gradually build interaction data for collaborative filtering.
* Consider **hybrid models** once data accumulates.

---

### If your dataset contains a high percentage of duplicate entries, why might this affect your model’s performance?

* Inflates importance of certain patterns → **model bias**.
* Leads to **overfitting**, as the model memorizes repeated samples.
* Misleads performance metrics.

**Solution**: Deduplicate based on relevant features before training.

---

### Why might deploying a model trained on cloud GPUs fail to perform well on edge devices?

* Model may be too **large or compute-intensive** for edge hardware.
* Potential issues:

  * Latency
  * Power constraints
  * Hardware incompatibility

**Solutions**:

* Apply **model compression** (e.g., pruning, quantization).
* Use **lightweight architectures** (e.g., MobileNet, TinyML models).

---

### You’re working with a time-series dataset where sudden spikes occur. Why might traditional smoothing techniques fail, and what alternatives would you use?

* Traditional smoothing (e.g., moving average) assumes gradual changes.
* Sudden spikes may be **real events** (e.g., sales, outages) and get smoothed out.

**Alternatives**:

* **Robust smoothing methods** like **exponential smoothing with anomaly detection**.
* Use **state-space models** (e.g., Kalman Filter).
* Incorporate **event indicators** or **change point detection algorithms**.


---


### Why is mean squared error (MSE) preferred over mean absolute error (MAE) in some cases, and vice versa?

* **MSE** penalizes **larger errors more heavily** due to squaring, making it suitable when large errors are particularly undesirable (e.g., in finance).

* **MSE is differentiable everywhere**, making it more compatible with gradient-based optimization.

* **MAE** treats all errors equally, which makes it **more robust to outliers**. It's preferred when outliers are common and shouldn't overly influence the model.

**In practice**:

* Use **MSE** when large deviations are critical.
* Use **MAE** when you want a more robust, median-like behavior.

---

### If a model has 95% accuracy, does it mean it’s a good model? Why or why not?

Not necessarily. High accuracy can be **misleading in imbalanced datasets**.

Example:

* If 95% of emails are non-spam, a model predicting "not spam" every time gets 95% accuracy — but **zero recall** for spam.

**Better approach**: Use metrics like **precision, recall, F1-score**, or **ROC-AUC**, especially for **imbalanced classification** problems.

---

### Why do deep learning models perform better with large amounts of data but struggle with small datasets?

* Deep learning models have **millions of parameters** and need large datasets to learn meaningful patterns and **avoid overfitting**.
* With small datasets, they tend to **memorize** the training data, leading to poor generalization.

**Alternatives for small data**:

* Use **simpler models** (e.g., decision trees, logistic regression).
* Apply **transfer learning** or **data augmentation** to mitigate the problem.

---

### What happens if you remove all hidden layers from a neural network?

You get a **single-layer model**, equivalent to:

* **Logistic regression** for classification.
* **Linear regression** for regression.

Without hidden layers, the model becomes **linear** and can't capture complex, non-linear patterns in the data.

---

### Can a model with high bias ever outperform a model with low bias?

Yes — especially on **small or noisy datasets**.

* High-bias models (e.g., linear models) may **generalize better** if the data is noisy or limited.
* Low-bias models (e.g., deep nets) can **overfit**, especially if there's insufficient training data.

**Bias-variance tradeoff** is key: sometimes, **more bias leads to lower generalization error**.

---

### If the probability of rain tomorrow is 80%, does that mean it will rain for 80% of the day? Why or why not?

No. A probability of 80% means there's an **80% chance that it will rain at any point during the day**, **not** that it will rain **for 80% of the time**.

It reflects **likelihood**, not **duration**.

---

### Why can a small dataset sometimes give misleading statistical results?

* **High variance**: Small samples can fluctuate significantly.
* **Outliers** can skew results.
* The **law of large numbers** hasn’t taken effect, so estimates (mean, variance, etc.) may be unreliable.

Always treat small-sample statistics with caution and **report uncertainty (e.g., confidence intervals)**.

---

### If you flip a fair coin 10 times and get 7 heads, does that mean the coin is biased? Why or why not?

Not necessarily.

* This result is within the range of **random variation** for a fair coin.
* 7 heads in 10 flips has a probability of \~17%, which is not uncommon.

To conclude bias, you’d need **more trials and statistical testing** (e.g., chi-square or binomial test).

---

### If a dataset has a mean of 50 and a median of 30, what does that suggest about its distribution?

It suggests the distribution is **right-skewed** (positively skewed).

* **Mean > Median** indicates the presence of **high-value outliers** pulling the mean up.

This asymmetry is a sign to look deeper into data spread and possibly apply transformations.

---

### Why is it incorrect to say that a p-value of 0.03 means there is a 97% probability the null hypothesis is false?

Because a **p-value is not the probability the null is false**.

* A p-value of 0.03 means: *If the null hypothesis were true*, the probability of observing data this extreme (or more) is **3%**.
* It **does not** tell us the probability that the null hypothesis is actually false — that would require **Bayesian analysis**.


---



### Difference between K-Means and DBSCAN

K-Means is a **centroid-based** clustering algorithm that partitions data into a fixed number of clusters (you must predefine *k*). It works best with **spherical and equally sized clusters** and is sensitive to outliers.

DBSCAN (Density-Based Spatial Clustering of Applications with Noise), on the other hand, is a **density-based** algorithm. It groups together points that are closely packed (based on a distance and minimum number of points), and marks low-density points as noise. DBSCAN doesn’t require the number of clusters to be specified and can find arbitrarily shaped clusters.

**Key differences:**

* K-Means assumes convex clusters; DBSCAN can handle non-convex shapes.
* K-Means is sensitive to outliers; DBSCAN naturally handles them.
* K-Means requires *k*; DBSCAN uses `eps` and `minPts`.

---

### Difference between Factor Analysis and PCA

Both are dimensionality reduction techniques, but they differ in **purpose and assumptions**.

PCA (Principal Component Analysis) is **variance-focused**. It transforms data into components that explain the maximum variance. It’s purely mathematical and doesn’t assume any underlying data generation model.

Factor Analysis, however, is **model-based** and focuses on explaining the observed variables in terms of a few **latent factors** plus noise. It assumes that variability in data is due to these hidden factors and tries to uncover them.

**Key differences:**

* PCA captures maximum variance; Factor Analysis models latent structure.
* PCA includes both common and unique variance; Factor Analysis aims to isolate only the shared (common) variance.
* PCA is often used for feature reduction; Factor Analysis is used for understanding underlying constructs (like in psychology or social sciences).



---

###  How does Boosting work?

Boosting is an **ensemble learning technique** that combines multiple weak learners to create a strong predictive model. The most common form is **gradient boosting**.

The process is **sequential**—each new model is trained to **correct the errors of the previous ones**. For example, the first tree might misclassify certain data points. The second tree is trained on the **residuals** or errors from the first, and so on. Each model gets added with a weight to form the final prediction.

The core idea is to **focus more on hard-to-learn examples**. In algorithms like AdaBoost, misclassified samples get **higher weights** in the next iteration. In gradient boosting (e.g., XGBoost), we optimize a loss function using **gradient descent in function space**—each tree is a step in minimizing the loss.

This method is powerful because it:

* Reduces bias and variance.
* Works well with tabular data.
* Can overfit if not regularized (via learning rate, tree depth, etc.).

---

### How to fine-tune XGBoost

Tuning XGBoost is about balancing **bias, variance, and training efficiency**.

Key hyperparameters:

1. **Learning rate (`eta`)**: Controls how much each tree contributes. Smaller values (e.g., 0.01–0.1) improve generalization but need more trees.
2. **`n_estimators`**: Number of boosting rounds. Often tuned along with `eta`.
3. **`max_depth`**: Controls model complexity. Larger depth = higher variance.
4. **`subsample` & `colsample_bytree`**: Random row and feature sampling. Reduces overfitting and improves generalization.
5. **`gamma`**: Minimum loss reduction required to make a further split.
6. **`reg_alpha` and `reg_lambda`**: L1 and L2 regularization.

**Steps to tune**:

* Use **RandomSearchCV** or **Optuna** for efficient search.
* Start with tree depth, learning rate, and estimators.
* Fix these, then tune regularization and sampling.

Monitor:

* **Training vs validation error**.
* Use **early stopping** to avoid overfitting.

---

###  **Explain WoE, IV, and VIF**

These are classic **feature evaluation and selection tools**, especially in credit scoring and regression:


* **Weight of Evidence (WoE)**: Converts categorical or binned numerical features into continuous values by comparing proportions of good vs bad outcomes:

  $$
  \text{WoE} = \log\left(\frac{\text{\% of Goods in bin}}{\text{\% of Bads in bin}}\right)
  $$

  Makes variables more linear and model-friendly.



* **Information Value (IV)**: Measures predictive power of a feature to separate binary outcomes. Calculated using Weight of Evidence bins.

  * IV < 0.02: Not useful
  * IV 0.1–0.3: Medium predictive
  * IV > 0.3: Strong predictive

  $$
  IV_i = (\text{\% of Goods in bin} - \text{\% of Bads in bin}) \times WoE_i\\

  IV = \sum IV_i
  $$


* **Variance Inflation Factor (VIF)**: Measures multicollinearity in regression. VIF > 5 or 10 indicates high collinearity.

  $$
  \text{VIF}_i = \frac{1}{1 - R_i^2}
  $$

  High VIF can inflate standard errors and make model unstable.

---

### ✅ **Gini Impurity vs Entropy**

Both are used in decision trees to measure **node impurity**, i.e., how mixed the classes are.

* **Gini Impurity**:

  $$
  G = 1 - \sum p_i^2
  $$

  Measures the probability that a randomly chosen sample would be incorrectly classified if randomly labeled according to the class distribution. It’s faster to compute and often used in CART.

* **Entropy**:

  $$
  H = -\sum p_i \log_2 p_i
  $$

  Comes from information theory. Measures average information (or surprise) required to identify class labels.

**Comparison**:

* Both aim to split the data in a way that increases purity.
* Gini is more computationally efficient. Use this when speed is more important
* Entropy tends to penalize impurity slightly more. Use this when the interpretability or theory is important. 

In practice, they often result in similar trees.

---

### **Time Complexity of Linear Regression**

**Closed-form (Normal Equation)**:

$$
\hat{\beta} = (X^TX)^{-1}X^Ty
$$

* Computing $X^TX$: $\mathcal{O}(nd^2)$
* Inverting $X^TX$: $\mathcal{O}(d^3)$
* Total: $\mathcal{O}(nd^2 + d^3)$

**Gradient Descent**:

* Each iteration: $\mathcal{O}(nd)$
* For $k$ iterations: $\mathcal{O}(knd)$

**When to use which?**

* Closed-form is fast for small datasets.
* Gradient descent is scalable and used when $d$ is large.

---


###  How does weight sharing differ between CNNs and RNNs?

**Weight sharing** reduces the number of parameters, enabling models to generalize better and learn efficiently.

* In **CNNs**, weight sharing happens **spatially**. A filter (kernel) slides across different parts of an image using the **same weights**. This enables CNNs to detect patterns (like edges or textures) regardless of their position in the input. So, if a cat’s ear appears at the top-left or bottom-right of an image, the same filter detects it.

* In **RNNs**, weight sharing happens **temporally**. The **same set of weights** is applied across all time steps in the sequence. This helps the RNN generalize to sequences of varying length. For instance, in language modeling, the weights that process the first word are reused for the second, third, and so on.

> ✅ So: CNNs reuse weights across **space**, RNNs reuse them across **time**.

This difference also affects the model’s **parallelizability** and the **types of patterns** they learn.

---

###  Why do RNNs suffer from vanishing gradients more than CNNs?

RNNs process sequences **one step at a time**, passing hidden states forward through potentially **hundreds of time steps**. During backpropagation, the chain rule multiplies many small derivatives (from sigmoid or tanh activations), causing gradients to shrink **exponentially** as we go backward in time — this is the **vanishing gradient problem**.

CNNs don’t have this issue because:

* They have a **fixed number of layers** (not unrolled over time).
* Their filters apply locally and don’t pass hidden states across long distances.
* They often use **ReLU**, which doesn’t squash gradients like sigmoid/tanh.

So, **depth over time** is the root cause for vanishing gradients in RNNs.

---

### Why are LSTMs and GRUs better than vanilla RNNs?

**Vanilla RNNs** pass the hidden state from one step to the next, but suffer from:

* Vanishing gradients
* Difficulty remembering long-term dependencies

**LSTMs (Long Short-Term Memory)** and **GRUs (Gated Recurrent Units)** solve this by introducing **gating mechanisms**:

* **Forget gate**: Decides what to discard.
* **Input gate**: Decides what new information to add.
* **Output gate**: Decides what to expose to the next step.

These mechanisms let the model **control information flow**, allowing gradients to survive across long time spans. GRUs are a simpler version of LSTMs with fewer gates but often perform comparably.

In short:

* Better memory
* Faster training
* Mitigated vanishing gradients

That’s why they are widely used in NLP and time-series.

---

### Can CNNs be used on sequences?

Yes — CNNs, especially **1D CNNs**, are often used on sequential data like text, audio, and time series.

Instead of sliding filters over 2D image grids, 1D CNNs slide over **time steps or tokens**. They’re useful for:

* Capturing **local temporal patterns** (e.g., n-grams in text)
* Fast, parallel training (unlike RNNs)
* Fixed-size output from variable-length input (with pooling)

However, CNNs struggle with **long-range dependencies**, which RNNs, LSTMs, or Transformers handle better.

📌 Use 1D CNNs when:

* You care more about **speed** and **local patterns**
* You want to **avoid sequential bottlenecks** in RNNs

---

### How do CNNs reduce dimensionality while preserving features?

CNNs reduce input dimensionality using:

1. **Convolution with stride > 1**: Skips some positions during the filter application, reducing width/height.
2. **Pooling (usually max pooling)**: Downsamples by summarizing nearby values (e.g., taking the max over a 2×2 region).

These steps:

* Reduce computation
* Prevent overfitting
* Keep **salient features** like edges or textures intact

As a result, CNNs compress the input while retaining the most **informative patterns**.

---

### What’s the role of padding in CNNs?

Padding adds **extra pixels (usually zeros)** around the input before applying convolution. It serves three purposes:

1. **Preserve spatial dimensions**: Without padding, every convolution reduces size. Padding keeps input and output the same size (called *same padding*).
2. **Help with edge detection**: Without padding, edge pixels get ignored more often. Padding ensures filters can cover corners too.
3. **Control model depth**: Padding allows us to stack more layers without shrinking input too quickly.

No padding = smaller output. Padding = controlled output size and better feature extraction.

---

### Why can’t CNNs handle temporal dependencies like RNNs?

CNNs process data **locally** — their receptive field is limited unless you stack many layers or use dilation. So they **can’t inherently remember past inputs** unless explicitly designed (e.g., with memory or recurrence).

RNNs, on the other hand:

* Maintain a **hidden state** over time
* Are designed for **temporal sequence modeling**

CNNs can model **short-term** dependencies well (with wide filters), but for **long-term** or **order-sensitive tasks**, RNNs or Transformers are more appropriate.

Example: CNN can classify a sequence of words as “positive” or “negative”, but for language generation or translation, you’d need temporal memory (RNN/Transformer).

---

### Parallelization: CNNs vs RNNs

* **CNNs** are **highly parallelizable**. Each convolution over pixels or tokens can happen simultaneously. That's why CNNs train fast on GPUs.
* **RNNs** are **sequential by nature**. Each step depends on the output of the previous, so training can’t be parallelized easily.

This makes CNNs more efficient and scalable. RNNs are slower, especially on long sequences, unless modified (e.g., with truncated backprop or parallel variants like QRNN).

---

### How to use Callback to stop learning in a neural network

In deep learning, especially with Keras or PyTorch Lightning, you can use a **callback** to stop training early if the model stops improving — this is called **EarlyStopping**.

In Keras:

```python
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(
    monitor='val_loss',     # Watch validation loss
    patience=3,             # Wait 3 epochs without improvement
    restore_best_weights=True
)

model.fit(X_train, y_train, validation_data=(X_val, y_val), callbacks=[early_stop])
```

This:

* Saves training time
* Avoids overfitting
* Restores the best weights before validation loss started increasing

You can also monitor **accuracy**, **F1**, or **custom metrics**. Other useful callbacks include `ReduceLROnPlateau`, `ModelCheckpoint`, and `TensorBoard`.


---

### How do you evaluate RAG output?

**RAG (Retrieval-Augmented Generation)** combines retrieval (e.g., from a vector DB) with generative models (e.g., LLMs). Evaluating it involves **two key components**:

1. **Retrieval Quality**:

   * 📌 **Precision\@k** or **Recall\@k**: Do the top-k retrieved documents contain the relevant information?
   * 🧠 **Embedding similarity** between question and retrieved chunks
   * ✅ **Human judgment**: Are the retrieved docs contextually relevant?

2. **Generation Quality**:

   * **Factual correctness**: Is the answer grounded in the retrieved content? You can use:

     * 🔍 **Faithfulness metrics** like **QAG (Question-Answer Generation)** or **FEVER score**
     * 📚 Compare with **reference answers** using:

       * **ROUGE** (recall-focused), **BLEU** (precision), **BERTScore** (semantic)
   * **Answer coverage**: Did the model answer the entire question?
   * **Conciseness** and **fluency**

3. **Holistic Metrics** (emerging):

   * **Ragas**: Combines retrieval relevance, answer faithfulness, and fluency
   * **TruLens**, **G-Eval**, and other LLM-based evaluation tools are also used

So, RAG evaluation is multi-dimensional and can’t be fully captured by traditional metrics alone. Human + automated evaluation is often necessary.

---

### How to evaluate text summarization using LLMs

There are two main types of summarization:

* **Extractive**: Picking key sentences
* **Abstractive**: Generating new phrasings

#### 🔍 Common evaluation metrics:

1. **ROUGE (Recall-Oriented Understudy for Gisting Evaluation)**

   * Measures n-gram overlap with reference summaries
   * ROUGE-1, ROUGE-2 (unigram, bigram), ROUGE-L (longest common subsequence)

2. **BLEU** (usually for translation, but still used)

3. **BERTScore**

   * Uses BERT embeddings to compare semantic similarity
   * Better for **abstractive summaries**

4. **LLM-as-a-judge**: Use GPT or similar to evaluate based on:

   * **Relevance**: Does it capture all key points?
   * **Factual consistency**: Any hallucinations?
   * **Fluency**: Is it readable and coherent?

#### 🧠 Advanced metrics:

* **QAGS**: Ask factual questions about the summary and see if it answers correctly.
* **SummaC**: Measures factual consistency by comparing entailment between source and summary.

#### 🚨 Note:

* ROUGE doesn't capture meaning well for abstractive summaries.
* So pair automatic metrics with **manual or LLM-based evaluation** for high-stakes tasks.

---

### Explain PEFT (Parameter-Efficient Fine-Tuning)

**PEFT** refers to techniques that fine-tune **only a small subset of a large model's parameters**, rather than the whole thing. This reduces **compute cost**, **memory usage**, and **storage**, while still achieving near full fine-tuning performance.

The most common PEFT method is **LoRA (Low-Rank Adaptation)**:

* Instead of updating full weight matrices, LoRA inserts **small trainable rank-decomposition matrices**.
* Only a few million parameters are trained, while the base model remains frozen.
* These modifications are merged at inference time.

Other PEFT methods:

* **Prefix Tuning**: Learn a trainable prefix of key/value vectors in attention layers.
* **Adapter Layers**: Add small feed-forward layers between transformer layers.
* **BitFit**: Only fine-tune bias terms.

#### ✅ Why use PEFT?

* 🚀 Finetune billion-parameter models on consumer hardware
* 🔄 Reuse the same base model for many tasks (just switch adapters)
* 🧪 Enables fast experimentation and deployment

**Popular in**: LLM adaptation (e.g., QLoRA), multi-task learning, edge deployment.


