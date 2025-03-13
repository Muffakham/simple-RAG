# RAG-based Question Answering System

This project demonstrates a Retrieval Augmented Generation (RAG) system for answering questions based on a provided document. It leverages LangChain for building the pipeline and utilizes a Sentence Transformer model for embedding generation and FAISS for efficient vector storage. The system answers questions by retrieving relevant document sections and providing a concise answer based on the retrieved context.

## Components

### 1. Document Loading and Splitting:

- **`UnstructuredMarkdownLoader`:** Loads the content from a markdown file (`medicare_comparison.md`).
- **`RecursiveCharacterTextSplitter`:** Splits the document into smaller chunks for efficient processing.

### 2. Embedding and Vector Store:

- **`SentenceTransformerEmbeddings`:** Generates embeddings for each document chunk using the "all-MiniLM-L6-v2" model.
- **`FAISS`:** Creates a vector store to index and search the document embeddings efficiently.

### 3. Retrieval:

- **`FAISS.as_retriever`:** Uses the FAISS vector store to retrieve relevant document chunks based on the query's embedding. A similarity score threshold is used to filter results.

### 4. Question Answering:

- **`ChatOpenAI`:** Uses the OpenAI GPT model (`gpt-4o`) as the language model for generating answers.
- **`CustomRetrievalQA`:** A custom class that manages the interaction between the retriever and the language model. It checks if relevant documents are found before invoking the LLM to answer the query.

## Running the Project

1. **Prepare Input Files:**
   - save markdown file named `medicare_comparison.md` containing the document to be used for question answering in the project folder.
   - save a JSON file named `queries.json` with an array of objects, where each object has a "text" field representing the question to be answered in the project folder.

2. **Add OPEN AI KEY**
   - The base LLM used is open AI's gpt-4o. An openAI key is required to run the project.
   - Either add the key in the `rag.py` file
   - Or, add the key as an environment variable `OPENAI_API_KEY`

2. **install the dependencies:**
   - run the `pip isntall -r requirements.txt` command to install all the required dependencies.

3. **run the app.py file**
   
4. **Results**:
   - Results are stored in the `results.json` file in the project folder.
