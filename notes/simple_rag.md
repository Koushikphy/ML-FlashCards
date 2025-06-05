

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


