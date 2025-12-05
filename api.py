from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from PyPDF2 import PdfReader
from dotenv import load_dotenv
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import os


load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
app = FastAPI()

# allow JS front-end
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

llm = ChatOpenAI(
    model="openai/gpt-4o-mini",
    openai_api_base="https://openrouter.ai/api/v1",
    api_key=API_KEY,
    temperature=0.3
)

def read_pdf(pdf_path):
    text = ""
    pdf = PdfReader(pdf_path)
    for page in pdf.pages:
        extracted = page.extract_text()
        if extracted:
            text += extracted
    return text

raw_text = read_pdf("restaurant_menu.pdf")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=120,
)
chunks = splitter.split_text(raw_text)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)
db = FAISS.from_texts(chunks, embeddings)

memory = ConversationBufferMemory(
    return_messages=True,
    memory_key="chat_history",
    output_key="answer"
)
chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=db.as_retriever(),
    memory=memory,
)
class Question(BaseModel):
    query: str

@app.post("/chat")
async def chat(q: Question):
    response = chain.invoke({"question": q.query})
    return {"answer": response["answer"]}
