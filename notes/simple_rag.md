

### 🔧 Install Dependencies (commented out)

```python
#! pip install chromadb==0.5.5 langchain-chroma==0.1.2 langchain==0.2.11 langchain-community==0.2.10 langchain-text-splitters==0.2.2 langchain-groq==0.1.6 transformers==4.43.2 sentence-transformers==3.0.1 unstructured==0.15.0 unstructured[pdf]==0.15.0
```

* This line (commented) installs necessary Python libraries:

  * **LangChain packages** for LLM orchestration.
  * **ChromaDB** as a vector store.
  * **HuggingFace Transformers** and SentenceTransformers for embedding text.
  * **Unstructured** for parsing PDFs or other document formats.
  * **Groq** for connecting with Groq’s LLM API.

---

### 🧱 Imports and API Key Setup

```python
import os
```

* Standard Python library for interacting with the operating system (e.g., setting environment variables).

```python
from langchain.document_loaders import UnstructuredFileLoader
```

* Loads unstructured documents (like PDFs, Word files) using LangChain.

```python
from langchain_text_splitters import CharacterTextSplitter
```

* Tool to split documents into manageable chunks of text based on character count.

```python
from langchain_community.embeddings import HuggingFaceEmbeddings
```

* Loads pre-trained embedding models (e.g., `all-MiniLM-L6-v2`) from HuggingFace.

```python
from langchain_chroma import Chroma
```

* Integration of LangChain with ChromaDB to store and search vector embeddings.

```python
from langchain_groq import ChatGroq
```

* LangChain interface to interact with **Groq's LLM API**.

```python
from langchain.chains import RetrievalQA
```

* Combines a retriever and an LLM to create a **Retrieval-Based QA chain**.

```python
os.environ["GROQ_API_KEY"] = "your_groq_api_key"
```

* Sets the Groq API key to authorize LLM access. Replace `"your_groq_api_key"` with your actual key.

---

### 📄 Load Document

```python
loader = UnstructuredFileLoader("attention_is_all_you_need.pdf")
documents = loader.load()
```

* Uses `UnstructuredFileLoader` to parse and load content from the PDF.
* `documents` is a list of `Document` objects containing the content and metadata.

---

### ✂️ Split Document into Chunks

```python
text_splitter = CharacterTextSplitter(
    chunk_size=2000,
    chunk_overlap=400
)
```

* Creates a splitter that breaks text into chunks of 2000 characters, with 400-character overlaps between chunks.
* Helps preserve context in downstream QA.

```python
texts = text_splitter.split_documents(documents)
```

* Applies the splitter to break the loaded PDF into overlapping text segments (LangChain `Document` objects).

---

### 🧠 Convert Text Chunks to Vectors (Embeddings)

```python
embeddings = HuggingFaceEmbeddings()
```

* Loads the default HuggingFace embedding model (usually SentenceTransformers).

```python
persist_directory = "doc_db"
```

* Specifies the directory where ChromaDB will persist the vector store.

```python
vectordb = Chroma.from_documents(
    documents=texts,
    embedding=embeddings,
    persist_directory=persist_directory
)
```

* Converts each chunk to a vector using the embedding model.
* Stores them in a Chroma vector database in the `doc_db` folder.

---

### 🔍 Set Up Retrieval + LLM

```python
retriever = vectordb.as_retriever()
```

* Converts the vector store into a retriever object, which fetches relevant chunks for any given query.

```python
llm = ChatGroq(
    model="llama-3.1-70b-versatile",
    temperature=0
)
```

* Connects to the **Groq-hosted LLaMA 3.1 model (70B parameters)**.
* `temperature=0` ensures deterministic output (no randomness).

---

### 🔗 Build the Retrieval-QA Chain

```python
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)
```

* Constructs a RetrievalQA pipeline:

  * Retrieves relevant text chunks from ChromaDB.
  * Feeds them into the LLM (`llm`).
  * `chain_type="stuff"`: simply concatenates all retrieved chunks before sending to the model.
  * `return_source_documents=True`: includes the source chunks used in the answer.

---

### 🧠 Ask a Question!

```python
query = "What is the model architecture discussed in this paper?"
response = qa_chain.invoke({"query": query})
```

* Asks the question using the `invoke()` method.
* LangChain:

  * Uses `retriever` to fetch top relevant chunks.
  * Sends them along with the query to the LLM.
  * Gets a natural language answer.

```python
print(response)
```

* Prints the response, including the answer and optionally the source documents.

---

### 🔁 Summary of Flow

1. Load PDF.
2. Split it into overlapping chunks.
3. Convert each chunk to a vector and store in ChromaDB.
4. Set up retriever and connect it to Groq’s LLM.
5. Ask a natural language question.
6. Retrieve relevant chunks and get a coherent answer.





-----
### Evaluating RAG output


Evaluating the output of a **Retrieval-Augmented Generation (RAG)** system is different from evaluating a traditional generative model because you're evaluating **both retrieval and generation components**. An end-to-end evaluation should include metrics that assess:

1. **Retrieval quality** – Did the model retrieve relevant and useful documents?
2. **Faithfulness and factual accuracy** – Did the generated output stay faithful to the retrieved documents?
3. **Relevance/Answer Quality** – Is the answer relevant, complete, and understandable?
4. **Latency or efficiency** – (Optional in production) How quickly and efficiently the RAG model operates.

---

## 🔧 Setup: Simple End-to-End Example

Let’s suppose we are building a RAG system for a **Q\&A bot** over a **Wikipedia-like knowledge base**.

### 🔹 Step 1: Sample Data

```python
query = "What are the health benefits of green tea?"
```

### 🔹 Step 2: Retrieved Documents (Top-3 Passages from Vector DB)

```python
retrieved_docs = [
    "Green tea contains polyphenols that have been shown to reduce inflammation and help fight cancer.",
    "It is rich in antioxidants called catechins, which may help prevent cell damage.",
    "Green tea has small amounts of caffeine and L-theanine, which together may improve brain function."
]
```

### 🔹 Step 3: Generated Answer

```python
generated_answer = (
    "Green tea offers multiple health benefits. It is rich in antioxidants like catechins "
    "which help prevent cell damage. It also contains polyphenols that reduce inflammation and "
    "may help fight cancer. Additionally, the combination of caffeine and L-theanine can improve brain function."
)
```

---

## ✅ Evaluation Strategy

We now evaluate the output using **automatic metrics** and **human criteria**.

---

### 📌 1. **Retrieval Evaluation**

#### 🔸 Metric: Recall\@k or Precision\@k

Check if the relevant ground truth document is in top-k retrieved.

```python
# Assume ground truth doc:
ground_truth_doc = "Green tea contains polyphenols that reduce inflammation and help fight diseases."

recall_at_3 = any(ground_truth_doc in doc for doc in retrieved_docs)  # Output: True
```

Or use **embedding similarity** (cosine similarity between ground truth doc and retrieved docs).

---

### 📌 2. **Faithfulness Evaluation (Factual Consistency)**

We ask:

* Are all factual claims in the generated answer **supported by the retrieved documents**?
* Is there any **hallucination**?

#### 🔸 Option 1: LLM-as-a-Judge (automated fact-checking)

Use a model like GPT-4 or another LLM with this prompt:

```python
prompt = f"""Given the retrieved documents: {retrieved_docs}
Evaluate whether the following answer is fully supported by them.
Answer: "{generated_answer}"
Is it fully supported? If not, highlight hallucinated parts.
"""
```

#### 🔸 Option 2: Faithfulness Score

Some tools offer automated factuality scoring, such as:

* **FactCC**, **DAE**, or **QAGS**
* Or use OpenAI's `text-davinci-003` or GPT-4 to rate faithfulness from 1 to 5.

---

### 📌 3. **Answer Quality (Relevance, Fluency, Completeness)**

Ask:

* Does the answer address the question?
* Is it coherent and easy to read?

#### 🔸 Metric: ROUGE / BLEU / METEOR (if ground-truth exists)

```python
from rouge_score import rouge_scorer
scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
scorer.score("Green tea has health benefits such as reducing inflammation and improving brain function.",
             generated_answer)
```

#### 🔸 Metric: BERTScore (semantic similarity)

```python
from bert_score import score
P, R, F1 = score([generated_answer], [reference_answer], lang="en")
```

#### 🔸 Metric: Human/LLM-based scoring

Prompt example:

```python
prompt = f"""
Evaluate the following answer for the question: "{query}"
Answer: "{generated_answer}"
Criteria:
- Relevance (Does it answer the question?)
- Completeness
- Clarity
Rate each on a scale of 1 to 5.
"""
```

---

### 📌 4. **Optional: Hallucination Detection**

Use automatic hallucination detection (e.g., via prompting an LLM):

```python
prompt = f"""Compare the generated answer to the retrieved passages.
Highlight any parts of the answer that are not supported by the passages.

Answer: {generated_answer}
Retrieved: {retrieved_docs}
"""
```

---

## 🔚 Summary of Metrics You Can Use

| Component      | Metric                            | Method                                   |
| -------------- | --------------------------------- | ---------------------------------------- |
| Retrieval      | Recall\@k, Precision\@k           | Ground truth check, similarity           |
| Faithfulness   | QA-check, LLM judge, FactCC       | Model evaluation or LLM prompt           |
| Answer Quality | ROUGE, BLEU, BERTScore, LLM judge | If reference answer exists               |
| Fluency        | Perplexity, LLM judge             | GPT-4, Claude scoring                    |
| Hallucination  | LLM-based comparison              | GPT prompt comparing output vs retrieved |

---

## 🧪 Toolkits and Frameworks for Evaluation

* **RAGAS** (Retrieval-Augmented Generation Assessment)
* **LangChain Evaluators** (`langchain.evaluation`)
* **TruLens** – For logging and evaluating LLM outputs
* **LlamaIndex evals**
* **Promptfoo** – For prompt evaluation and testing

