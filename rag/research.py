import os
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI
import os

# Set your Google API key as an environment variable
os.environ["GOOGLE_API_KEY"] = "KEY"

# Sample documents for the vector store
documents = [
    "Cats are known for their playful nature and agility.",
    "Dogs are loyal companions and often called man's best friend.",
    "Birds can fly and are known for their beautiful songs.",
]

# Convert texts to LangChain Document objects
docs = [Document(page_content=text) for text in documents]

# Initialize embeddings with an open-source model
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Create a Chroma vector store
vectorstore = Chroma.from_documents(docs, embeddings)

# Initialize Google Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0.7, top_p=0.85)


qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vectorstore.as_retriever()
)

# Interactive question-answering loop
print("Type 'exit' to quit.")
while True:
    query = input("Ask a question: ")
    if query.lower() == "exit":
        break
    # Show the retrieved document
    retrieved_docs = vectorstore.similarity_search(query, k=1)
    print("Retrieved document:", retrieved_docs[0].page_content)
    # Generate and display the answer
    result = qa_chain(query)
    print("Answer:", result['result'])