from custom_qa_chain import CustomQA
from rag import RAG

doc_file = "medicare_comparison.md"
query_file = "queries.json"
rag_tool = RAG(doc_file, query_file)
rag_tool.run()