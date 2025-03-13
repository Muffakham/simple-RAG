from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.schema import BaseRetriever
from langchain import PromptTemplate

class CustomQA():

    """
    custom QA chain.
    Checks if the retrriever is able to match any docs.
    If so, the LLM is called, else, No LLM is called.
    A message is returned.
    """
    def __init__(self, llm_model, retriever: BaseRetriever):
        
        
        template = """
          System: You are a intelligent question answering bot, that provides short answers to questions.
            
          Instructions: 
          1. Make Use the following context to to answer the user query.
          2. Make sure the answer is short and brief.
          3. Use the context to generate the answer.

          Context: {context}

          Human: {inputs}

          Assistant:"""

        prompt_template = PromptTemplate(
              input_variables=["inputs", "context"],
              template=template,
        )

        
        self.qa_chain = create_stuff_documents_chain(llm=llm_model, prompt = prompt_template)         
        self.retriever = retriever
        self.llm_model = llm_model

    def run(self, inputs):
        
        question = inputs["query"]
        retrieved_docs = self.retriever.get_relevant_documents(question)

        if not retrieved_docs:
            print("No matching documents found. Skipping LLM call.")
            return {"result": "I couldn't find any information about that.", "source_documents": []} 
        else:
            asnwer = self.qa_chain.invoke({
                "inputs": question,
                "context": retrieved_docs
            })
            response = {}
            response["result"] = asnwer
            response["source_documents"] = retrieved_docs
            return response
