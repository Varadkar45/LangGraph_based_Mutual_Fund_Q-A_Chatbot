import os
import streamlit as st
from datetime import datetime
from RAG_with_Langgraph import (
    extract_text_from_multiple_files,
    split_text,
    create_documents,
    create_retriever,
    setup_user_query_chain,
    ChatState,
    create_langgraph_app,
    file_paths,
)
st.set_page_config(layout="wide")

# Optional: fix torch warning
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# # Heading (centered)
st.sidebar.markdown("<h2 style='text-align: left; font-size: 28px;'>🤖Q&A Chatbot based on HDFC Mutual Fund Factsheets</h2>", unsafe_allow_html=True)
st.sidebar.markdown("<h4 style='text-align: left; font-size: 18px'>The chatbot can answer questions related to Mutual Fund Factsheets, including details about fund performance📊, investment strategies, risk factors, NAV (Net Asset Value), asset allocation, historical returns, and expense ratios. It helps investors understand and analyze mutual fund data efficiently. 🚀</h4>", unsafe_allow_html=True)

# st.sidebar.markdown("<h2>📊 Q&A Chatbot based on HDFC Mutual Fund Factsheets</h2>", unsafe_allow_html=True)
# st.sidebar.text_input("Ask a question:", key="user_question")
# st.markdown(
#     """
#     <style>
#         .sticky-header {
#             position: fixed;
#             top: 0;
#             left: 0;
#             width: 100%;
#             padding: 10px;
#             z-index: 100;
#         }
#     </style>
#     <div class="sticky-header">
#         <h2>📊 Q&A Chatbot based on HDFC Mutual Fund Factsheets</h2>
#         <h4>🤖 Ask questions about HDFC Mutual Funds</h4>
#     </div>
#     """,
#     unsafe_allow_html=True
# )


# Initialize LangGraph App
@st.cache_resource
def initialize_chatbot():
    content = extract_text_from_multiple_files(file_paths)
    chunks = split_text(content)
    documents = create_documents(chunks)
    retriever = create_retriever(documents)
    user_query_chain = setup_user_query_chain(retriever)
    app = create_langgraph_app(user_query_chain)
    return app, user_query_chain

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

app, user_query_chain = initialize_chatbot()

# Show chat history (Q card left, A card right below with timestamps)
if st.session_state.chat_history:
    st.markdown("---")
    st.markdown("<h4 style='text-align: center;'>📝 Conversation History</h4>", unsafe_allow_html=True)
    for i in range(0, len(st.session_state.chat_history), 2):
        timestamp = st.session_state.timestamps[i // 2] if i // 2 < len(st.session_state.timestamps) else ""
        st.markdown("""
            <div style='margin-bottom: 20px;'>
                <div style='background-color: #000000; padding: 10px; border-radius: 10px; width: 60%; text-align: left;'>
                    <strong>Q:</strong> {question}<br>
                    <small style='color: red;'>{timestamp}</small>
                </div>
                <div style='height: 10px;'></div>
                <div style='background-color: #000000; padding: 10px; border-radius: 10px; width: 60%; float: right; text-align: right;'>
                    <strong>A:</strong> {answer}<br>
                    <small style='color: red;'>{timestamp}</small>
                </div>
                <div style='clear: both;'></div>
            </div>
        """.format(
            question=st.session_state.chat_history[i],
            answer=st.session_state.chat_history[i + 1],
            timestamp=timestamp
        ), unsafe_allow_html=True)

# Spacer before input box
st.markdown("<div style='height: 0px;'></div>", unsafe_allow_html=True)

# User input (bottom section)
with st.container():
    question = st.text_input("Ask a question about HDFC Mutual Funds:", key="user_question")

    if question:
        chat_history = [{"role": "user", "content": q} if i % 2 == 0 else {"role": "assistant", "content": q}
                        for i, q in enumerate(st.session_state.chat_history)]

        state = ChatState(chat_history=chat_history, question=question, answer="")
        result = app.invoke(state)

        st.session_state.chat_history.append(question)
        st.session_state.chat_history.append(result["answer"])
        st.session_state.timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

        st.markdown(f"**Answer:** {result['answer']}")