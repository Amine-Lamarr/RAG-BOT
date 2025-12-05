from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
# from langchain.vectorstores import faiss
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os 

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")

# Ensure API Key exists
if not API_KEY:
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_base="https://openrouter.ai/api/v1",
    api_key=API_KEY, 
    temperature=0.7
)
# print(llm.invoke("hi").content)

# PDFS reader
def Reading(pdfs):
    text = ""  
    for pdf in pdfs:  
        try:
            with open(pdf, "rb") as file:  
                pdf_reader = PdfReader(file)  
                for page in pdf_reader.pages:  
                    extracted = page.extract_text()
                    if extracted:
                        text += extracted
        except Exception as e:
            print(f"Error reading {pdf}: {e}")
    return text

# chunks
def Chunking(text):
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "\n"], 
        chunk_size = 500, 
        chunk_overlap = 120
    )
    texts = splitter.split_text(text)
    return texts

# get vectorstore
def VectStor(chunks):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectstore = FAISS.from_texts(chunks, embeddings)
    return vectstore

# retreiver
def Retriever(vectstore):
    memory = ConversationBufferMemory(
        memory_key="chat_history", 
        return_messages=True,
        output_key="answer"
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectstore.as_retriever(),
        memory=memory,
        return_source_documents=False
    )
    return conversation_chain
    
def Agent(query, chain):
    results = chain.invoke({"question": query})
    return results['answer'] 

pdf = ["restaurant_menu.pdf"] 
if not os.path.exists(pdf[0]):
    print(f"File not found: {pdf[0]}")
else:
    files = Reading(pdf)

    if not files:
        print("No text extracted from PDF. Script stopped.")
    else:
        texts = Chunking(files)
        vectstor = VectStor(texts)
        retriever = Retriever(vectstor)
        
        qst = "how much is a chicken tagine ?"
        response = Agent(qst, retriever)
        print(f"\nAnswer: {response}")

