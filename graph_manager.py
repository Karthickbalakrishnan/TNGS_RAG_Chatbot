import os
import pickle
import networkx as nx
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List
import matplotlib.pyplot as plt

load_dotenv()

GRAPH_FILE = "scheme_graph.pkl"
URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

# --- 1. ENTITY EXTRACTION SCHEMA ---
class SchemeEntity(BaseModel):
    scheme_name: str = Field(description="Name of the government scheme")
    department: str = Field(description="Issuing department name")
    beneficiary_group: str = Field(description="Target beneficiaries e.g. Women, Unemployed Youth, Farmers")
    benefit_details: str = Field(description="Core financial or social benefit provided")

class SchemeExtractionList(BaseModel):
    schemes: List[SchemeEntity]

def extract_entities_from_webpage() -> List[dict]:
    print("🌐 Scraping TN Government webpage for entity extraction...")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(URL, headers=headers)
    if response.status_code != 200:
        print("Failed to fetch webpage.")
        return []
        
    soup = BeautifulSoup(response.text, 'html.parser')
    raw_text = soup.get_text(separator=" ", strip=True)
    
    print("🤖 Extracting structured entities using LLM...")
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    parser = JsonOutputParser(pydantic_object=SchemeExtractionList)
    
    prompt = PromptTemplate(
        template="Extract structured scheme entities from the following Tamil Nadu government webpage text.\n{format_instructions}\nText:\n{text}",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    try:
        result = chain.invoke({"text": raw_text[:15000]})
        return result.get("schemes", [])
    except Exception as e:
        print(f"Extraction error: {e}")
        return []

# --- 2. GRAPH STORAGE & PERSISTENCE ---
def build_and_save_graph():
    schemes = extract_entities_from_webpage()
    if not schemes:
        print("No schemes extracted.")
        return
        
    G = nx.DiGraph()
    
    for item in schemes:
        dept = item.get("department", "Social Welfare Department").strip()
        scheme = item.get("scheme_name", "").strip()
        beneficiary = item.get("beneficiary_group", "General Public").strip()
        benefits = item.get("benefit_details", "As per scheme norms").strip()
        
        # Add Nodes with Metadata
        G.add_node(dept, type="Department")
        G.add_node(scheme, type="Scheme", benefits=benefits)
        G.add_node(beneficiary, type="Beneficiary")
        
        # Add Directed Edges (Relationships)
        G.add_edge(dept, scheme, relation="OFFERS")
        G.add_edge(scheme, beneficiary, relation="TARGETS")
        
    # Save graph to disk using Pickle persistence
    with open(GRAPH_FILE, "wb") as f:
        pickle.dump(G, f)
        
    print(f"✅ Graph successfully built and saved to '{GRAPH_FILE}'!")
    print(f"📊 Total Nodes: {G.number_of_nodes()} | Total Edges: {G.number_of_edges()}")
    
    # Automatically export visualization after building
    export_graph_to_image(G)

def load_graph() -> nx.DiGraph:
    if os.path.exists(GRAPH_FILE):
        with open(GRAPH_FILE, "rb") as f:
            return pickle.load(f)
    print("⚠️ Graph file not found. Building a new graph...")
    build_and_save_graph()
    return load_graph()

# --- 3. GRAPH VISUALIZATION & EXPORT ---
def export_graph_to_image(G: nx.DiGraph):
    print("🎨 Generating visual graph export (PDF/PNG)...")
    plt.figure(figsize=(14, 10))
    pos = nx.spring_layout(G, k=0.3, seed=42)
    
    node_colors = []
    for node, data in G.nodes(data=True):
        ntype = data.get("type", "Unknown")
        if ntype == "Department":
            node_colors.append("#ff7f0e")  # Orange
        elif ntype == "Scheme":
            node_colors.append("#1f77b4")  # Blue
        else:
            node_colors.append("#2ca02c")  # Green
            
    nx.draw_networkx_nodes(G, pos, node_size=700, node_color=node_colors, alpha=0.9)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.5, edge_color="gray", arrows=True)
    
    labels = {node: (node[:25] + "..." if len(node) > 25 else node) for node in G.nodes()}
    nx.draw_networkx_labels(G, pos, labels=labels, font_size=8, font_family="sans-serif")
    
    plt.title("Tamil Nadu Government Schemes Knowledge Graph (GraphRAG)", fontsize=14, fontweight="bold")
    plt.axis("off")
    plt.tight_layout()
    
    plt.savefig("tn_schemes_knowledge_graph.png", dpi=300, bbox_inches="tight")
    plt.savefig("tn_schemes_knowledge_graph.pdf", bbox_inches="tight")
    plt.close()
    print("✅ Exported files: 'tn_schemes_knowledge_graph.png' & 'tn_schemes_knowledge_graph.pdf'")

# --- 4. GRAPH QUERYING (MULTI-HOP TRAVERSAL) ---
def query_graph(G: nx.DiGraph, search_term: str) -> List[dict]:
    results = []
    search_lower = search_term.lower()
    
    for node, data in G.nodes(data=True):
        if data.get("type") == "Scheme":
            if search_lower in node.lower() or search_lower in data.get("benefits", "").lower():
                depts = [u for u, v, d in G.in_edges(node, data=True) if d.get("relation") == "OFFERS"]
                beneficiaries = [v for u, v, d in G.out_edges(node, data=True) if d.get("relation") == "TARGETS"]
                
                results.append({
                    "scheme": node,
                    "department": depts[0] if depts else "TN Government",
                    "beneficiary": ", ".join(beneficiaries) if beneficiaries else "General Public",
                    "benefits": data.get("benefits")
                })
    return results

if __name__ == "__main__":
    build_and_save_graph()