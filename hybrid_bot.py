import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Graph context retrieval helper
def fetch_graph_context(query: str) -> str:
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    graph_data = []
    
    cypher_query = """
    MATCH (d:Department)-[:OFFERS]->(s:Scheme)-[:TARGETS]->(b:Beneficiary)
    WHERE toLower(s.name) CONTAINS toLower($term) 
       OR toLower(b.group) CONTAINS toLower($term)
       OR toLower(d.name) CONTAINS toLower($term)
    RETURN d.name AS dept, s.name AS scheme, s.benefits AS benefits, b.group AS target
    LIMIT 3
    """
    
    # Extract search keyword or default to general match
    search_term = query.split()[-1] if len(query.split()) > 0 else ""
    
    with driver.session() as session:
        results = session.run(cypher_query, term=search_term)
        for record in results:
            graph_data.append(f"Department: {record['dept']} | Scheme: {record['scheme']} | Target Group: {record['target']} | Benefits: {record['benefits']}")
            
    driver.close()
    return "\n".join(graph_data) if graph_data else "No specific graph relation found."

# Combined Hybrid Prompt Template
hybrid_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
[ROLE]
You are an expert AI Government Scheme Navigator for Tamil Nadu using a Hybrid Knowledge Graph + Vector RAG architecture.

[VECTOR CONTEXT]
{vector_context}

[KNOWLEDGE GRAPH RELATIONS]
{graph_context}

[TASK]
Answer the user's question accurately combining both contexts. Address the user by name ({user_name}). Use clear bullet points for benefits and eligibility. If data is missing, state it clearly.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

def run_hybrid_chatbot():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    vector_retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, streaming=True)
    
    print("\n" + "="*60)
    print("🤖 TN Government Hybrid GraphRAG Navigator Initialized")
    print("="*60)
    
    user_name = input("Hello! What is your name?\n> ").strip().capitalize()
    chat_history = []
    
    while True:
        query = input(f"\n{user_name}: ")
        if query.lower() == 'exit':
            break
        if not query.strip():
            continue
            
        print("\nAssistant: ", end="", flush=True)
        
        try:
            # 1. Retrieve Vector Chunks
            vector_docs = vector_retriever.invoke(query)
            vector_text = "\n\n".join([doc.page_content for doc in vector_docs])
            
            # 2. Retrieve Graph Relations
            graph_text = fetch_graph_context(query)
            
            # 3. Format prompt
            prompt_formatted = hybrid_chat_prompt.invoke({
                "vector_context": vector_text,
                "graph_context": graph_text,
                "chat_history": chat_history,
                "user_name": user_name,
                "question": query
            })
            
            full_response = ""
            for chunk in llm.stream(prompt_formatted):
                content = chunk.content
                print(content, end="", flush=True)
                full_response += content
                
            print("\n" + "-"*50)
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=full_response))
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    run_hybrid_chatbot()