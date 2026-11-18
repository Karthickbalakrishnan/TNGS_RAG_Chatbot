import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Import NetworkX graph manager functions
from graph_manager import load_graph, query_graph

# Load environment variables
load_dotenv()

# Page Configuration
st.set_page_config(
    page_title="TN Govt Scheme Navigator - GraphRAG", 
    page_icon="🏛️", 
    layout="centered"
)

st.title("🏛️ Tamil Nadu Government Scheme Navigator (GraphRAG + Guardrails)")
st.caption("Secure, hallucination-free AI assistant powered by Hybrid Vector Search and NetworkX Knowledge Graphs.")

# --- 1. LOAD VECTOR STORE & NETWORKX KNOWLEDGE GRAPH ---
@st.cache_resource
def load_rag_and_graph():
    # Load Chroma Vector Store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if not os.path.exists("./chroma_db"):
        return None, None, None
        
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    # Load or build persistent NetworkX Knowledge Graph
    graph_db = load_graph()
    
    # Initialize LLM with streaming enabled
    llm = ChatOpenAI(model="gpt-4o", temperature=0.2, streaming=True)
    
    return retriever, graph_db, llm

retriever, graph_db, llm = load_rag_and_graph()

if not retriever or graph_db is None:
    st.error("Error: Database or Knowledge Graph not found! Please run `python3 ingest.py` and `python3 graph_manager.py` first.")
    st.stop()

# --- 2. INPUT GUARDRAIL FUNCTION ---
def guardrail_check(query: str) -> bool:
    """Blocks off-topic queries (weather, sports, personal coding help, etc.)"""
    off_topic_keywords = ["weather", "cricket", "movie", "python code", "recipe", "stock price", "bitcoin"]
    query_lower = query.lower()
    if any(word in query_lower for word in off_topic_keywords):
        return False
    return True

# Initialize Session State
if "user_name" not in st.session_state:
    st.session_state.user_name = ""

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Ask for Name if not provided
if not st.session_state.user_name:
    with st.form("name_form"):
        st.subheader("👋 Welcome! Let's get acquainted.")
        name_input = st.text_input("Please enter your name:")
        submitted = st.form_submit_button("Start Navigating")
        if submitted and name_input.strip():
            st.session_state.user_name = name_input.strip().capitalize()
            st.rerun()
    st.stop()

st.success(f"Connected as **{st.session_state.user_name}**. Guardrails & NetworkX Knowledge Graph active!")

# --- SIDEBAR: KNOWLEDGE GRAPH INSPECTOR ---
with st.sidebar.expander("📊 Inspect Knowledge Graph"):
    st.write(f"**Total Graph Nodes:** {graph_db.number_of_nodes()}")
    st.write(f"**Total Graph Edges:** {graph_db.number_of_edges()}")
    
    st.markdown("### 🔗 Stored Relationships:")
    edge_count = 0
    for u, v, data in graph_db.edges(data=True):
        if edge_count < 15:  # Limit display for clean UI
            relation = data.get('relation', 'RELATES_TO')
            st.markdown(f"- **{u[:30]}...** ➔ *{relation}* ➔ **{v[:30]}...**")
            edge_count += 1
        else:
            st.markdown(f"_...and {graph_db.number_of_edges() - 15} more edges_")
            break
        
    st.markdown("### 📋 Scheme Metadata Sample:")
    scheme_count = 0
    for node, data in graph_db.nodes(data=True):
        if data.get("type") == "Scheme" and scheme_count < 5:
            st.markdown(f"**Scheme:** `{node}`")
            st.json(data)
            scheme_count += 1

if st.sidebar.button("Clear Chat History"):
    st.session_state.chat_history = []
    st.rerun()

# RTCFR System Prompt Template with Hybrid Context
rtcfr_graph_prompt = ChatPromptTemplate.from_messages([
    ("system", """
[ROLE]
You are a warm, empathetic, and expert AI Government Scheme Navigator for the Government of Tamil Nadu. Your job is to help citizens accurately discover welfare schemes, eligibility rules, and application procedures based exclusively on the provided context and NetworkX Knowledge Graph facts.

[VECTOR CONTEXT]
{vector_context}

[KNOWLEDGE GRAPH TRUTH (NETWORKX)]
{graph_context}

[TASK]
Answer the user's question accurately using ONLY the provided context and graph relations. 
- Address the user by name ({user_name}).
- If information is not present or contradicts the official database, state that you cannot verify the record to prevent hallucinations.

[RESPONSE FORMAT]
- Use clear headings and bullet points for eligibility, benefits, and application steps.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_hybrid_graph_context(query: str):
    # Query NetworkX graph using multi-hop lookup
    matches = query_graph(graph_db, query)
    if not matches:
        return "Verified standard government welfare guidelines apply."
    
    context_lines = []
    for m in matches:
        context_lines.append(
            f"Scheme: {m['scheme']} | Department: {m['department']} | Target: {m['beneficiary']} | Benefits: {m['benefits']}"
        )
    return "\n".join(context_lines)

# Render Chat History
for message in st.session_state.chat_history:
    role = "user" if isinstance(message, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(message.content)

# Chat Input & Processing with Guardrails
if query := st.chat_input(f"Ask a question, {st.session_state.user_name}..."):
    
    # 1. Apply Input Guardrail
    if not guardrail_check(query):
        st.session_state.chat_history.append(HumanMessage(content=query))
        blocked_msg = "⚠️ **Guardrail Notice:** I am specialized exclusively in Tamil Nadu government welfare schemes and programs. I cannot answer queries outside this domain. Please ask about welfare benefits, eligibility, or departments!"
        st.session_state.chat_history.append(AIMessage(content=blocked_msg))
        st.rerun()

    st.session_state.chat_history.append(HumanMessage(content=query))
    with st.chat_message("user"):
        st.markdown(query)
        
    # 2. Retrieve Hybrid Context (Vector Search + NetworkX Graph Traversal)
    with st.chat_message("assistant"):
        with st.spinner("Verifying with NetworkX Knowledge Graph & Vector Search..."):
            docs = retriever.invoke(query)
            vector_text = format_docs(docs)
            graph_text = get_hybrid_graph_context(query)
            
            prompt_formatted = rtcfr_graph_prompt.invoke({
                "vector_context": vector_text,
                "graph_context": graph_text,
                "chat_history": st.session_state.chat_history[:-1],
                "user_name": st.session_state.user_name,
                "question": query
            })
        
        def stream_generator():
            for chunk in llm.stream(prompt_formatted):
                yield chunk.content
                
        response_content = st.write_stream(stream_generator())
        st.session_state.chat_history.append(AIMessage(content=response_content))