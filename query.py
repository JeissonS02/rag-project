import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# Initialize Pinecone
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index_name = os.getenv("PINECONE_INDEX")

# Initialize embeddings (same model used in ingest)
embeddings = HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
)

# Connect to existing vector store
vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# Initialize Groq LLM
llm = ChatGroq(
    model="llama-3.1-8b-instant",
    temperature=0.2
)

# Prompt template
prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant.

Use the following retrieved context to answer the question.
If the answer is not contained in the context, say that you don't know.

Context:
{context}

Question:
{question}
"""
)

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# Ask user question
question = input("Ask a question: ")

# Retrieve relevant documents
docs = retriever.invoke(question)
context = format_docs(docs)

# Create chain
chain = prompt | llm

# Generate response
response = chain.invoke({
    "context": context,
    "question": question
})

print("\nAnswer:\n")
print(response.content)
