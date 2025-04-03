import fitz
import pandas as pd
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import TextLoader
from langchain.docstore.document import Document
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langgraph.graph import StateGraph, END, START
from dataclasses import dataclass

# LLM setup
groq_api_key = "your_api_key"
llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-70b-8192")

# File paths
file_path_1 = "HDFC_MF_Factsheet_October_2024.pdf"
file_path_2 = "HDFC_MF_Factsheet_September_2024_0.pdf"
file_path_3 = "HDFC_MF_Factsheet_August_2024.pdf"

file_paths = [file_path_1, file_path_2, file_path_3]

# Extraction, splitting, embedding
def extract_text_from_multiple_files(file_paths):
    def extract_text_from_file(file_path):
        if file_path.endswith('.pdf'):
            doc = fitz.open(file_path)
            content = "".join(page.get_text() for page in doc)
            return content
        elif file_path.endswith('.xlsx'):
            return pd.read_excel(file_path).to_string(index=False)
        elif file_path.endswith('.csv'):
            return pd.read_csv(file_path).to_string(index=False)
        return ""
    return "\n".join(extract_text_from_file(fp) for fp in file_paths if extract_text_from_file(fp))

def split_text(content):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
    return text_splitter.split_text(content)

def create_documents(chunks):
    return [Document(page_content=chunk) for chunk in chunks]

def create_retriever(documents):
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore.as_retriever()

# Chain
def setup_user_query_chain(retriever):
    prompt_template = """Role: You are a helpful AI assistant that answers user questions based on the provided documents.
Objective: Provide clear, accurate, and helpful responses using the available information..

    data: {context}
    Previous Conversations:
    {chat_history}

    QUESTION: {question}

    Instructions:
    - Just provide the answer, nothing else.
    - Ensure responses are concise and relevant.
    - Support answers with key data insights when necessary.
    - If data is insufficient, respond with 'The available data does not contain sufficient information to answer this question.'
    """

    PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question", "chat_history"])
    memory = ConversationBufferMemory(memory_key="chat_history", input_key="question", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )

# LangGraph state
@dataclass
class ChatState:
    chat_history: list
    question: str
    answer: str

def create_langgraph_app(user_query_chain):
    def process_query(state: ChatState):
        chat_history = user_query_chain.memory.load_memory_variables({}).get("chat_history", [])
        response = user_query_chain.invoke({"question": state.question, "chat_history": chat_history})
        state.answer = response['answer'].replace("**", "").strip()
        user_query_chain.memory.save_context({"question": state.question}, {"answer": state.answer})
        return state

    graph = StateGraph(ChatState)
    graph.add_node("query_processing", process_query)
    graph.add_edge(START, "query_processing")
    graph.add_edge("query_processing", END)
    return graph.compile()
