
### Why is subword tokenization (e.g., Byte-Pair Encoding) useful in handling rare words in NLP tasks?

* Rare or OOV words are broken into **subword units**, reducing vocabulary size while preserving meaning.
* Allows **compositional representation** of new or morphologically rich words.
* Balances between word-level and character-level tokenization.

---

### Why might overfitting be a bigger issue in NLP tasks compared to tabular data?

* **High dimensionality**: Text is typically sparse and high-dimensional.
* **Semantic redundancy**: Models can memorize patterns instead of learning general semantics.
* **Small labeled datasets**: Especially in supervised NLP, labeled text is expensive.

---

### If you’re training an NLP model, why might smaller batch sizes lead to better generalization for text data?

* Smaller batches introduce **more gradient noise**, which can act as a regularizer.
* Helps **escape sharp minima** and leads to better generalization, especially with noisy textual data.
* Often beneficial when data is semantically diverse.

---

### Why do transformer models like GPT require positional encoding, and how does it work?

* Transformers lack recurrence, so they can’t inherently model sequence order.
* **Positional encodings** (sinusoidal or learned) are added to input embeddings to give the model a sense of token **position**.
* Enables the model to capture **order-dependent relationships**.

---

### Why might text summarization models struggle with long documents, and how would you overcome this?

* Transformers have **quadratic complexity** with input length.
* **Context window** is limited; they may miss important sections.

**Solutions:**

* Use **long-context transformers** (e.g., Longformer, BigBird).
* Apply **hierarchical models** or **extractive summarization** as a preprocessing step.
* Use **sliding windows** or chunk-and-merge techniques.



---


### Why might removing stopwords sometimes hurt model performance instead of improving it?

Removing stopwords can hurt performance when:

* **Stopwords carry syntactic or semantic importance** in the task. For example, in sentiment analysis, words like *"not"*, *"never"*, or *"no"* completely reverse sentiment.
* Some stopwords contribute to **intent** or **meaning**, especially in tasks like question answering, dialogue modeling, or text entailment.
* It may disrupt **contextual relationships** in models that rely on word co-occurrence or sequence (e.g., n-gram models, transformers).

**Takeaway**: Stopword removal isn’t always beneficial — it's task-dependent.

---

### If two sentences have the same words but in a different order, will a bag-of-words model treat them as the same? Why or why not?

Yes, a **Bag-of-Words (BoW)** model will treat them as the same, because:

* BoW only considers the **frequency** of words, **ignoring order**.
* For example, *"Dog bites man"* and *"Man bites dog"* will have identical vector representations, even though their meanings differ drastically.

This limitation is one reason why BoW is often replaced by models that preserve **word order** (e.g., RNNs, transformers).

---

### Why can’t we just use TF-IDF for deep learning-based NLP models?

TF-IDF is:

* **Sparse and high-dimensional**.
* **Static**: It lacks contextual understanding—"bank" in "river bank" vs. "savings bank" gets the same vector.
* Not suitable for models like neural networks that expect **dense, low-dimensional, and continuous representations**.

Instead, deep learning uses **word embeddings** (e.g., Word2Vec, GloVe) or **contextual embeddings** (e.g., BERT) that capture **semantic meaning** and **contextual usage**.

---

### How does a chatbot understand the intent behind a sentence?

Chatbots understand intent using:

* **Intent classification models**: Trained on labeled utterances (e.g., "book a flight", "check weather").
* **Embeddings**: Sentences are converted into vectors using models like BERT, and then classified.
* Often part of a **Natural Language Understanding (NLU)** pipeline, which includes:

  * **Intent recognition**
  * **Entity extraction**
  * Optional **dialogue state tracking**

Fine-tuning on task-specific data or using frameworks like **Rasa**, **Dialogflow**, or **transformers** is common.

---

### Why is it difficult for AI models to understand sarcasm?

Sarcasm is hard for AI because:

* It often involves **implicit meaning**, contradiction between **literal words and actual intent**.
* Requires **contextual understanding**, tone, speaker intent, and sometimes **world knowledge** or **social cues**.
* Sarcastic text may resemble non-sarcastic text syntactically.

**Example**: “Oh great, another Monday morning meeting” — literally sounds positive, but is often sarcastic.

**Approaches to improve understanding**:

* Use **contextual embeddings** (e.g., BERT + sentiment cues).
* Include **user history**, **dialogue context**, or even **multimodal data** (like audio for tone).
* Train on **sarcasm-labeled datasets** (e.g., Twitter sarcasm corpus).

---
