import networkx as nx
import requests
from bs4 import BeautifulSoup
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

class SchemeEntity(BaseModel):
    scheme_name: str = Field(description="Name of the government scheme")
    department: str = Field(description="Issuing department name")
    beneficiary_group: str = Field(description="Target beneficiaries e.g. Women, Unemployed Youth, Farmers")
    benefit_details: str = Field(description="Core financial or social benefit provided")

class SchemeExtractionList(BaseModel):
    schemes: List[SchemeEntity]

def build_in_memory_graph():
    url = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(url, headers=headers)
    soup = BeautifulSoup(response.text, 'html.parser')
    raw_text = soup.get_text(separator=" ", strip=True)
    
    # Extract entities using LLM
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
        schemes = result.get("schemes", [])
    except Exception:
        schemes = []
        
    # Build NetworkX Graph
    G = nx.DiGraph()
    for item in schemes:
        dept = item.get("department", "Social Welfare Department")
        scheme = item.get("scheme_name")
        beneficiary = item.get("beneficiary_group", "General Public")
        benefits = item.get("benefit_details", "As per scheme norms")
        
        # Add nodes and edges
        G.add_node(dept, type="Department")
        G.add_node(scheme, type="Scheme", benefits=benefits)
        G.add_node(beneficiary, type="Beneficiary")
        
        G.add_edge(dept, scheme, relation="OFFERS")
        G.add_edge(scheme, beneficiary, relation="TARGETS")
        
    return G

def query_knowledge_graph(G, keyword: str):
    results = []
    keyword_lower = keyword.lower()
    for node, data in G.nodes(data=True):
        if data.get("type") == "Scheme" and (keyword_lower in node.lower() or keyword_lower in data.get("benefits", "").lower()):
            # Find incoming department and outgoing beneficiaries
            depts = [u for u, v, d in G.in_edges(node, data=True) if d.get("relation") == "OFFERS"]
            beneficiaries = [v for u, v, d in G.out_edges(node, data=True) if d.get("relation") == "TARGETS"]
            results.append({
                "scheme": node,
                "department": depts[0] if depts else "TN Government",
                "beneficiary": ", ".join(beneficiaries),
                "benefits": data.get("benefits")
            })
    return results