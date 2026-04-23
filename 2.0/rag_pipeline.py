import os
import re
from dotenv import load_dotenv, find_dotenv
from typing import TypedDict, List
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage, AIMessage
from langgraph.graph import StateGraph, START, END

load_dotenv(find_dotenv())

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FACTSHEETS_DIR = os.path.join(BASE_DIR, "factsheets")
CHROMA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "chroma_db")

PDF_FILES = [
    os.path.join(FACTSHEETS_DIR, "HDFC_MF_Factsheet_August_2024.pdf"),
    os.path.join(FACTSHEETS_DIR, "HDFC_MF_Factsheet_September_2024_0.pdf"),
    os.path.join(FACTSHEETS_DIR, "HDFC_MF_Factsheet_October_2024.pdf"),
]

SOURCES = [
    "HDFC_MF_Factsheet_August_2024.pdf",
    "HDFC_MF_Factsheet_September_2024_0.pdf",
    "HDFC_MF_Factsheet_October_2024.pdf",
]

embeddings = OllamaEmbeddings(model="snowflake-arctic-embed2")

groq_llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model_name="llama-3.3-70b-versatile"
)
ollama_llm = ChatOllama(model="gpt-oss:20b")
llm = groq_llm.with_fallbacks([ollama_llm])

REFORMULATE_PROMPT = PromptTemplate.from_template("""
Given the chat history below and a follow-up question, rewrite the follow-up question to be a
fully self-contained, standalone question that includes all necessary context (fund name, time period, etc.).

Rules:
- If the question refers to NAV or performance and does not specify a plan type, always add "Regular Plan Growth Option".
- If the question is already self-contained, return it unchanged.

Chat History:
{chat_history}

Follow-up Question: {query}

Standalone Question:
""")

ROUTER_PROMPT = PromptTemplate.from_template("""
You are a routing assistant for an HDFC Mutual Fund chatbot.

Classify the user message into one of two categories:
- "rag" — if the message is a question or request that needs information from HDFC Mutual Fund factsheets (NAV, fund performance, fund manager, expense ratio, portfolio holdings, AUM, returns, risk, etc.)
- "chat" — if the message is general conversation (greetings, introductions, thank you, small talk, anything not related to mutual funds)

Reply with a single word — "rag" or "chat".

User message: {query}
""")

GRADE_PROMPT = PromptTemplate.from_template("""
You are a relevance grader. Does the retrieved context contain enough information to answer the question?

Question: {query}

Retrieved Context:
{context}

Reply with a single word — "yes" if the context is relevant and useful, "no" if it is not.
""")

REWRITE_PROMPT = PromptTemplate.from_template("""
The current query did not retrieve useful documents from HDFC Mutual Fund factsheets.
Rewrite it using different, more specific keywords. Return ONLY the rewritten question as a single
sentence — no explanations, no alternatives, no formatting.

Original query: {query}

Rewritten question:
""")

ANSWER_PROMPT = PromptTemplate.from_template("""
You are a helpful assistant specializing in HDFC Mutual Fund factsheets covering August, September, and October 2024.

Instructions:
- Answer using only the provided context.
- For NAV queries, use the plan name exactly as stated in the context to identify the correct value.
  Regular Plan NAV is always lower than Direct Plan NAV for the same fund.
- Always mention which months the data is from.
- If data for a specific month is missing from the context, say so clearly.

Context:
{context}

Chat History:
{chat_history}

Question: {query}

Answer:
""")


def is_noise_page(content: str) -> bool:
    if "CONTENTS" in content and "PAGE NO." in content:
        return True
    if "Disclaimer:" in content and "views expressed herein" in content:
        return True
    lines = [l.strip() for l in content.split("\n") if l.strip()]
    if not lines:
        return True
    page_ref_count = sum(1 for l in lines if re.fullmatch(r"\d{1,3}(-\d{1,3})?", l))
    if len(lines) > 5 and page_ref_count / len(lines) > 0.25:
        return True
    return False


def extract_nav_summaries(pages: list, source_name: str) -> list:
    """Extract KEY FACTS NAV blocks from PDF pages and create dedicated summary documents.

    The factsheet layout is two-column: portfolio holdings fill the first ~5000 chars of the page,
    and the KEY FACTS section (fund name, plan names, NAV values) follows in the right column.
    This causes the NAV chunk to be dominated by portfolio text in similarity search, so we create
    clean, explicit NAV summary documents that rank correctly for NAV-related queries.
    """
    from langchain_core.documents import Document

    summaries = []
    plan_sequence = [
        "Regular Plan - Growth Option",
        "Regular Plan - IDCW Option",
        "Direct Plan - Growth Option",
        "Direct Plan - IDCW Option",
    ]

    for page in pages:
        content = page.page_content
        if "Regular Plan - Growth Option" not in content:
            continue

        lines = content.split("\n")
        for i, line in enumerate(lines):
            if line.strip() != "Regular Plan - Growth Option":
                continue
            # Verify the full 4-plan block follows
            if i + 3 >= len(lines):
                continue
            block_lines = [lines[i + j].strip() for j in range(4)]
            if block_lines != plan_sequence:
                continue

            # Collect the 4 NAV values from lines after the plan names
            nav_values = []
            j = i + 4
            while j < len(lines) and len(nav_values) < 4:
                val = lines[j].strip()
                if re.match(r'^\d+\.\d+$', val):
                    nav_values.append(val)
                elif val:
                    break  # non-numeric non-empty line breaks the sequence
                j += 1

            if len(nav_values) < 1:
                continue

            # Find nearest HDFC fund name in the preceding text
            preceding = "\n".join(lines[max(0, i - 40):i])
            fund_matches = list(re.finditer(r'HDFC[^\n(]+?(?:Fund|ETF|FoF)[^\n]*', preceding))
            if not fund_matches:
                continue  # skip if fund name not found — avoids "Unknown HDFC Fund" in responses
            fund_name = fund_matches[-1].group(0).strip()

            # Extract fund manager if present in preceding text
            mgr_match = re.search(
                r'FUND MANAGER[^\n]*\n([^\n]+)\n\(since ([^\)]+)\)', preceding
            )
            fund_manager_str = ""
            if mgr_match:
                fund_manager_str = f"\nFund Manager: {mgr_match.group(1).strip()} (since {mgr_match.group(2).strip()})"

            # Find the date in the following text
            following = "\n".join(lines[i:i + 30])
            date_match = re.search(r'As on (\w+ \d+, \d{4})', following)
            date_str = (date_match.group(1) if date_match
                        else source_name.replace("HDFC_MF_Factsheet_", "")
                                        .replace(".pdf", "").replace("_", " "))

            summary = (
                f"{fund_name} - NAV as of {date_str}\n"
                f"Regular Plan - Growth Option NAV: {nav_values[0] if len(nav_values) > 0 else 'N/A'}\n"
                f"Regular Plan - IDCW Option NAV: {nav_values[1] if len(nav_values) > 1 else 'N/A'}\n"
                f"Direct Plan - Growth Option NAV: {nav_values[2] if len(nav_values) > 2 else 'N/A'}\n"
                f"Direct Plan - IDCW Option NAV: {nav_values[3] if len(nav_values) > 3 else 'N/A'}"
                f"{fund_manager_str}\n"
                f"Source: {source_name}"
            )
            summaries.append(Document(
                page_content=summary,
                metadata={"source_file": source_name, "chunk_type": "nav_summary"}
            ))

    return summaries


def load_and_split_documents() -> list:
    from langchain_core.documents import Document

    splitter = RecursiveCharacterTextSplitter(chunk_size=2500, chunk_overlap=1000)
    all_chunks = []

    for path in PDF_FILES:
        source_name = os.path.basename(path)
        loader = PyMuPDFLoader(path)
        pages = loader.load()

        # Add dedicated NAV summary chunks (high-signal documents for NAV queries)
        nav_summaries = extract_nav_summaries(pages, source_name)
        all_chunks.extend(nav_summaries)
        print(f"  {source_name}: extracted {len(nav_summaries)} NAV summary chunks")

        clean_pages = [p.page_content for p in pages if not is_noise_page(p.page_content)]
        print(f"  {source_name}: {len(pages)} pages -> {len(clean_pages)} after filtering noise")

        full_text = "\n".join(clean_pages)
        doc = Document(page_content=full_text, metadata={"source_file": source_name})

        chunks = splitter.split_documents([doc])
        all_chunks.extend(chunks)

    return all_chunks


def get_vectorstore() -> Chroma:
    vectorstore = Chroma(
        collection_name="hdfc_factsheets",
        embedding_function=embeddings,
        persist_directory=CHROMA_DIR
    )
    if vectorstore._collection.count() == 0:
        print("Building vector store - this runs once and persists to disk...")
        docs = load_and_split_documents()
        vectorstore.add_documents(docs)
        print(f"Vector store built with {len(docs)} chunks.")
    else:
        print(f"Loaded existing vector store ({vectorstore._collection.count()} chunks).")
    return vectorstore


class ChatState(TypedDict):
    query: str
    standalone_query: str
    context: str
    chat_history: List
    answer: str
    retry_count: int
    is_relevant: bool
    route: str


CHAT_PROMPT = PromptTemplate.from_template("""
You are a friendly assistant for an HDFC Mutual Fund chatbot.
The user has sent a general message — respond naturally and warmly.
If relevant, let them know you can help with questions about HDFC Mutual Fund factsheets
(NAV, fund performance, fund manager, portfolio, returns, etc.).

Chat History:
{chat_history}

User message: {query}

Response:
""")


def build_graph(vectorstore: Chroma):

    def router_node(state: ChatState) -> ChatState:
        response = llm.invoke(ROUTER_PROMPT.format(query=state["query"]))
        route = "chat" if "chat" in response.content.strip().lower() else "rag"
        return {**state, "route": route, "standalone_query": state["query"]}

    def reformulate_node(state: ChatState) -> ChatState:
        if not state["chat_history"]:
            return {**state, "standalone_query": state["query"]}
        history_str = "\n".join([
            f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
            for msg in state["chat_history"]
        ])
        prompt = REFORMULATE_PROMPT.format(
            chat_history=history_str,
            query=state["query"]
        )
        response = llm.invoke(prompt)
        standalone = response.content.strip()
        return {**state, "standalone_query": standalone}

    def retrieve_node(state: ChatState) -> ChatState:
        all_docs = []
        for source in SOURCES:
            retriever = vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 4, "filter": {"source_file": source}}
            )
            all_docs.extend(retriever.invoke(state["standalone_query"]))

        context = "\n\n".join([
            f"[Source: {doc.metadata.get('source_file', 'Unknown')}]\n{doc.page_content}"
            for doc in all_docs
        ])
        return {**state, "context": context}

    def grade_relevance_node(state: ChatState) -> ChatState:
        response = llm.invoke(GRADE_PROMPT.format(
            query=state["standalone_query"],
            context=state["context"][:4000]
        ))
        is_relevant = "yes" in response.content.strip().lower()
        return {**state, "is_relevant": is_relevant}

    def rewrite_query_node(state: ChatState) -> ChatState:
        response = llm.invoke(REWRITE_PROMPT.format(query=state["standalone_query"]))
        new_query = response.content.strip()
        return {**state, "standalone_query": new_query, "retry_count": state["retry_count"] + 1}

    def generate_node(state: ChatState) -> ChatState:
        history_str = "\n".join([
            f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
            for msg in state["chat_history"]
        ])
        formatted_prompt = ANSWER_PROMPT.format(
            context=state["context"],
            chat_history=history_str,
            query=state["standalone_query"]
        )
        response = llm.invoke(formatted_prompt)
        answer = response.content.strip()
        updated_history = state["chat_history"] + [
            HumanMessage(content=state["query"]),
            AIMessage(content=answer)
        ]
        return {**state, "answer": answer, "chat_history": updated_history}

    def chat_node(state: ChatState) -> ChatState:
        history_str = "\n".join([
            f"Human: {msg.content}" if isinstance(msg, HumanMessage) else f"Assistant: {msg.content}"
            for msg in state["chat_history"]
        ])
        response = llm.invoke(CHAT_PROMPT.format(
            chat_history=history_str,
            query=state["query"]
        ))
        answer = response.content.strip()
        updated_history = state["chat_history"] + [
            HumanMessage(content=state["query"]),
            AIMessage(content=answer)
        ]
        return {**state, "answer": answer, "chat_history": updated_history}

    def route_after_router(state: ChatState) -> str:
        return state["route"]

    def route_after_grading(state: ChatState) -> str:
        if state["is_relevant"] or state["retry_count"] >= 2:
            return "generate"
        return "rewrite"

    graph = StateGraph(ChatState)
    graph.add_node("router", router_node)
    graph.add_node("reformulate", reformulate_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("grade", grade_relevance_node)
    graph.add_node("rewrite", rewrite_query_node)
    graph.add_node("generate", generate_node)
    graph.add_node("chat", chat_node)

    graph.add_edge(START, "router")
    graph.add_conditional_edges("router", route_after_router, {
        "rag": "reformulate",
        "chat": "chat"
    })
    graph.add_edge("reformulate", "retrieve")
    graph.add_edge("retrieve", "grade")
    graph.add_conditional_edges("grade", route_after_grading, {
        "generate": "generate",
        "rewrite": "rewrite"
    })
    graph.add_edge("rewrite", "retrieve")
    graph.add_edge("generate", END)
    graph.add_edge("chat", END)

    return graph.compile()


def save_graph_diagram(app):
    try:
        diagram_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "workflow_diagram.png")
        with open(diagram_path, "wb") as f:
            f.write(app.get_graph().draw_mermaid_png())
        print(f"Workflow diagram saved: {diagram_path}")
    except Exception as e:
        print(f"Could not save diagram: {e}")
