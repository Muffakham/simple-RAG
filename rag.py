from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import TextLoader, UnstructuredMarkdownLoader
from langchain_text_splitters import MarkdownHeaderTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI


import os

os.environ["OPENAI_API_KEY"] = "<< -- OPEN AI KEY -- >>"


class RAG:
  def __init__(self, doc_file, query_file):

    if not self.check_file_exists(doc_file):
      raise FileNotFoundError(f"File not found: {doc_file}")
    
    if not self.check_file_exists(query_file):
      raise FileNotFoundError(f"File not found: {query_file}")

    self.document = UnstructuredMarkdownLoader(doc_file).load()
    self.chunks = self.split_document(self.document)
    self.vector_store = self.create_vector_store(self.chunks)
    self.queries = load_json_file(query_file)
    self.results = []
    self.retriever = self.vector_store.as_retriever(search_type="similarity_score_threshold",
                                                    search_kwargs={"score_threshold": 0.2})
    self.llm = ChatOpenAI(model="gpt-4o")
    self.qa = CustomQA(llm_model=self.llm,
                                retriever=self.retriever)

  def check_file_exists(self, file_path):
    return os.path.exists(file_path)

  def split_document(self, document, chunk_size: int = 500, chunk_overlap: int = 0):
      """
      takes in the .MD file and splits it into chunks of text.
      input:
        file_path: string = path to the .MD file.
        chunk_size: integer = Number of chars in a single chunk.
        chunk_overlap: integer = Number of chars that are borrowed from the previous chunk.
      output:
        texts: list = list of chunks of text.

      """

      text_splitter = RecursiveCharacterTextSplitter(
          chunk_size=chunk_size, chunk_overlap=chunk_overlap
      )
      texts = text_splitter.split_documents(document)
      return texts

  def create_vector_store(self, chunks):
      """
      takes in the chunks of text and creates a FAISS vector store.
      By default, the model used is all-MiniLM-L6-v2.
      input:
        chunks: list = list of chunks of text.
      output:
        db: FAISS = vector store.
      """
      embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
      db = FAISS.from_documents(chunks, embeddings)
      return db

  def get_answer(self, query):
      """
      takes in a query and returns the answer.
      input:
        query: string = query to be answered.
      output:
        result: JSON = anser and source documents.
      """
      result = self.qa.run({"query": query})
      source_docs = result["source_documents"]
      
      return result

  def load_json_file(self, file_path):
    try:
        with open(file_path, 'r') as file:
            data = json.load(file)
            return data
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found.")
        return None
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in '{file_path}'.")
        return None

  def store_json_file(self):
    """
    stores the results in a JSON file
    stores a list of dictionaries in a JSON file.
    """

    with open("results.json", "w") as f:
        json.dump(self.results, f, indent=4)
  
  
  def run(self):
    """
    runs the RAG pipeline on the list of queries.
    stores the results in a JSON file.
    """

    results = []
    for query in self.queries:
        result = self.get_answer(query['text'])
        answer, source_documents = result["result"], result["source_documents"]
        query["answer"] = answer
        query["source_documents"] = [{"id": doc.id, "content": doc.page_content} for doc in source_documents]

        results.append(query)
    
    self.results = results
    self.store_json_file()
    return self.results
