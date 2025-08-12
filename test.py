from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
import os
from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain.chains import create_retrieval_chain

load_dotenv()


def load_doc(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=200,
        chunk_overlap=20
    )
    splitDocs = splitter.split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorStore = FAISS.from_documents(splitDocs,embedding=embeddings)
    return vectorStore

def create_chain(vectorStore):
    # Initialize the LLM
    llm = ChatOpenAI(model_name="openai/gpt-3.5-turbo",
                 base_url="https://openrouter.ai/api/v1",
                 openai_api_key=os.getenv("OPENROUTER_API_KEY"),
                 temperature=0.4,
                 max_tokens=500,
                 )
    
    # Prompt with context and user input
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant that answers questions based on the provided context."),
        ("human", "Answer the user's question:\nContext: {context}\nQuestion: {input}")
    ])
    
    chain = create_stuff_documents_chain(llm,prompt)
    
    retrieval = vectorStore.as_retriever(search_kwargs={"k" : 2})
    
    retrieval_chain = create_retrieval_chain(retrieval,chain)
    
    return retrieval_chain


vectorStore = load_doc("https://bewhoop.com/")
chain = create_chain(vectorStore)

response = chain.invoke({
    "input" : "How bewhoop works?"
})

print(response["answer"])