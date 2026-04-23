import streamlit as st
from datetime import datetime
from rag_pipeline import get_vectorstore, build_graph, save_graph_diagram

st.set_page_config(page_title="HDFC MF Chatbot", page_icon="📈", layout="wide")

# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    "<h2 style='text-align:left; font-size:26px;'>🤖 Q&A Chatbot based on HDFC Mutual Fund Factsheets</h2>",
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    "<h4 style='font-size:15px; font-weight:normal;'>"
    "Ask anything about HDFC Mutual Fund factsheets covering "
    "<b>August, September &amp; October 2024</b>.<br><br>"
    "Topics include fund performance 📊, NAV trends, investment strategies, "
    "risk factors, asset allocation, historical returns, expense ratios, "
    "fund managers, and more. 🚀"
    "</h4>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown("**Powered by**")
st.sidebar.markdown(
    "- LangGraph\n- ChromaDB\n- Groq LLaMA 3.3 70B\n- snowflake-arctic-embed2"
)


# ── Init ─────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Setting up the knowledge base...")
def initialize():
    vectorstore = get_vectorstore()
    app = build_graph(vectorstore)
    save_graph_diagram(app)
    return app


app = initialize()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "display_messages" not in st.session_state:
    st.session_state.display_messages = []
if "timestamps" not in st.session_state:
    st.session_state.timestamps = []

# ── Chat history display ──────────────────────────────────────────────────────
if st.session_state.display_messages:
    st.markdown(
        "<h4 style='text-align:center;'>📝 Conversation History</h4>",
        unsafe_allow_html=True,
    )
    st.markdown("---")
    for i, (q, a, ts) in enumerate(
        zip(
            st.session_state.display_messages[0::2],
            st.session_state.display_messages[1::2],
            st.session_state.timestamps,
        )
    ):
        st.markdown(
            f"""
        <div style='margin-bottom:24px;'>
            <div style='background-color:#1e1e2e; padding:12px 16px; border-radius:12px;
                        width:62%; text-align:left; border-left:4px solid #4a9eff;'>
                <strong style='color:#4a9eff;'>Q:</strong>
                <span style='color:#e0e0e0;'> {q}</span><br>
                <small style='color:#888;'>{ts}</small>
            </div>
            <div style='height:8px;'></div>
            <div style='background-color:#1e2e1e; padding:12px 16px; border-radius:12px;
                        width:62%; float:right; text-align:left; border-left:4px solid #4aff88;'>
                <strong style='color:#4aff88;'>A:</strong>
                <span style='color:#e0e0e0;'> {a}</span><br>
                <small style='color:#888;'>{ts}</small>
            </div>
            <div style='clear:both;'></div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ── Input ─────────────────────────────────────────────────────────────────────
st.markdown("<div style='height:20px;'></div>", unsafe_allow_html=True)
query = st.chat_input("Ask me anything about HDFC Mutual Funds...")

if query:
    # Show the question immediately so it doesn't disappear while LLM is thinking
    st.markdown(
        f"""
    <div style='margin-bottom:12px;'>
        <div style='background-color:#1e1e2e; padding:12px 16px; border-radius:12px;
                    width:62%; text-align:left; border-left:4px solid #4a9eff;'>
            <strong style='color:#4a9eff;'>Q:</strong>
            <span style='color:#e0e0e0;'> {query}</span>
        </div>
    </div>
    """,
        unsafe_allow_html=True,
    )

    with st.spinner("Thinking..."):
        result = app.invoke(
            {
                "query": query,
                "standalone_query": "",
                "context": "",
                "chat_history": st.session_state.chat_history,
                "answer": "",
                "retry_count": 0,
                "is_relevant": False,
                "route": "",
            }
        )

    answer = result["answer"]
    st.session_state.chat_history = result["chat_history"]
    st.session_state.display_messages.append(query)
    st.session_state.display_messages.append(answer)
    st.session_state.timestamps.append(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    st.rerun()
