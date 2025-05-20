import os

from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS

import config


class TariffRAGPipeline:
    def __init__(self, vectorstore_path="vectorstore"):
        self.vectorstore_path = vectorstore_path
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
        self.llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", api_key=config.GEMINI_API_KEY)

    def _load_vectorstore(self):
        """Load FAISS index from disk"""
        if not os.path.exists(self.vectorstore_path):
            raise FileNotFoundError(f"Vectorstore not found at {self.vectorstore_path}")

        return FAISS.load_local(self.vectorstore_path, self.embeddings, allow_dangerous_deserialization=True)

    def _create_qa_chain(self):
        """Create QA chain with custom prompt"""
        prompt = PromptTemplate.from_template(
            """
        You are a tariff calculation expert analyzing South African port fees.

        Use the following context to answer the question:
        {context}

        Question: {question}

        Instructions:
        1. Identify relevant tariff rules from the context
        2. Extract numerical values and formulas
        3. Show calculation steps
        4. Provide final amount with ZAR currency
        """
        )

        return RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=self._load_vectorstore().as_retriever(k=4),
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True,
        )

    def query_tariff(self, question: str):
        """Process a tariff-related query end-to-end"""
        qa_chain = self._create_qa_chain()
        query_result = qa_chain.invoke({"query": question})

        # Format output with source references
        formatted_result = {
            "answer": query_result["result"],
            "sources": [doc.metadata for doc in query_result["source_documents"]],
            "raw_chunks": [doc.page_content for doc in query_result["source_documents"]],
        }

        return formatted_result


# Example Usage
if __name__ == "__main__":
    # Initialize pipeline
    pipeline = TariffRAGPipeline()

    # Example query for tariff calculation
    query = """
    Calculate light dues for a vessel with:
    - Length Overall (LOA): 229.2 meters
    - Port: Durban
    - Vessel Type: Bulk Carrier
    - GT: 51,300
    """

    result = pipeline.query_tariff(query)

    print("Answer:\n", result["answer"])
    print("\nSources:\n", result["sources"])
    print("\nRetrieved Chunks:\n", "\n---\n".join(result["raw_chunks"]))
