    import os
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from graph_manager import load_graph, query_graph

load_dotenv()

# --- 1. DEFINE TEST DATASET (Golden Evaluation Set) ---
test_dataset = [
    {
        "question": "What is the Unemployed Youth Employment Generation Programme (UYEGP)?",
        "ground_truth": "UYEGP is a scheme managed by the MSME Department targeting unemployed youth to provide financial assistance and subsidies up to 25%."
    },
    {
        "question": "What financial benefits does the New Entrepreneur Cum Enterprise Development Scheme offer?",
        "ground_truth": "NEEDS provides a capital subsidy up to 25% and soft loans for first-generation entrepreneurs."
    },
    {
        "question": "What is the weather today in Chennai?",
        "ground_truth": "I cannot answer this as I am specialized exclusively in Tamil Nadu government welfare schemes."
    }
]

# --- 2. SETUP HYBRID RAG APP RUNNER ---
def run_rag_pipeline(query: str):
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 2})
    graph_db = load_graph()
    
    # Simple guardrail check
    off_topic = ["weather", "cricket", "movie", "python code"]
    if any(word in query.lower() for word in off_topic):
        return "⚠️ Guardrail Notice: I am specialized exclusively in Tamil Nadu government welfare schemes."
    
    # Retrieve docs & graph context
    docs = retriever.invoke(query)
    vector_text = "\n\n".join([d.page_content for d in docs])
    
    matches = query_graph(graph_db, query)
    graph_text = "\n".join([f"Scheme: {m['scheme']} | Dept: {m['department']} | Target: {m['beneficiary']} | Benefits: {m['benefits']}" for m in matches]) if matches else "Standard guidelines apply."
    
    prompt = PromptTemplate.from_template("""
You are an expert TN Government Scheme Navigator. Answer accurately using ONLY the context and graph facts.
Context: {vector_context}
Graph Truth: {graph_context}
Question: {question}
Answer:
""")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    chain = prompt | llm | StrOutputParser()
    
    response = chain.invoke({
        "vector_context": vector_text,
        "graph_context": graph_text,
        "question": query
    })
    return response

# --- 3. LLM-AS-A-JUDGE EVALUATION FUNCTION ---
def evaluate_response(question: str, context: str, response: str) -> dict:
    judge_llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    
    # Faithfulness Judge Prompt
    faithfulness_prompt = f"""
You are an AI evaluation expert. Evaluate if the response is strictly faithful to the provided context without introducing outside information or hallucinations.
Context: {context}
Response: {response}
Output ONLY a score of 1 (Faithful) or 0 (Unfaithful), followed by a brief reason in this format: Score: [0/1] - Reason: [text]
"""
    faith_result = judge_llm.invoke(faithfulness_prompt).content
    
    # Relevance Judge Prompt
    relevance_prompt = f"""
Evaluate if the response directly and accurately addresses the user's question.
Question: {question}
Response: {response}
Output ONLY a score of 1 (Relevant) or 0 (Irrelevant), followed by a brief reason in this format: Score: [0/1] - Reason: [text]
"""
    rel_result = judge_llm.invoke(relevance_prompt).content
    
    return {
        "faithfulness_evaluation": faith_result,
        "relevance_evaluation": rel_result
    }

# --- 4. EXECUTE EVALUATION SUITE ---
def run_evaluation():
    print("="*60)
    print("🧪 RUNNING RAG EVALUATION SUITE (LLM-as-a-Judge)")
    print("="*60)
    
    for i, test in enumerate(test_dataset, 1):
        print(f"\nTest Case {i}: {test['question']}")
        response = run_rag_pipeline(test["question"])
        print(f"🤖 Response:\n{response}\n")
        
        # Run judge evaluation
        eval_scores = evaluate_response(test["question"], test["ground_truth"], response)
        print(f"📊 Evaluation Scores:")
        print(f"  - {eval_scores['faithfulness_evaluation']}")
        print(f"  - {eval_scores['relevance_evaluation']}")
        print("-" * 50)

if __name__ == "__main__":
    run_evaluation()