import os
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from neo4j import GraphDatabase, TrustAll
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field
from typing import List

# Load environment variables from .env file
load_dotenv()

# Neo4j & OpenAI Credentials
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "password")

# Target URL for the Tamil Nadu Government Scheme List
URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

# Define Pydantic schema for structured entity extraction
class SchemeEntity(BaseModel):
    scheme_name: str = Field(description="Name of the government scheme")
    department: str = Field(description="Issuing department name")
    beneficiary_group: str = Field(description="Target beneficiaries e.g. Women, Farmers, Students")
    benefit_details: str = Field(description="Core financial or social benefit provided")

class SchemeExtractionList(BaseModel):
    schemes: List[SchemeEntity]

def extract_entities_with_llm(text: str) -> List[dict]:
    llm = ChatOpenAI(model="gpt-4o", temperature=0.0)
    parser = JsonOutputParser(pydantic_object=SchemeExtractionList)
    
    prompt = PromptTemplate(
        template="Extract structured scheme entities from the following Tamil Nadu government webpage text.\n{format_instructions}\nText:\n{text}",
        input_variables=["text"],
        partial_variables={"format_instructions": parser.get_format_instructions()},
    )
    
    chain = prompt | llm | parser
    try:
        result = chain.invoke({"text": text[:15000]} ) # Truncate token length safely
        return result.get("schemes", [])
    except Exception as e:
        print(f"Extraction error: {e}")
        return []

def populate_neo4j():
    # 1. Scrape Webpage Text
    print("Scraping TN Government webpage for graph ingestion...")
    headers = {"User-Agent": "Mozilla/5.0"}
    response = requests.get(URL, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch webpage. Status code: {response.status_code}")
        
    soup = BeautifulSoup(response.text, 'html.parser')
    raw_text = soup.get_text(separator=" ", strip=True)
    
    # 2. Extract Entities via LLM
    print("Extracting entities and relationships using LLM...")
    extracted_schemes = extract_entities_with_llm(raw_text)
    print(f"Extracted {len(extracted_schemes)} structured scheme entities.")
    
    # 3. Insert into Neo4j with TrustAll() to handle SSL certificates securely
    print("Connecting to Neo4j and saving entities...")
    driver = GraphDatabase.driver(
        NEO4J_URI, 
        auth=(NEO4J_USER, NEO4J_PASSWORD),
        trusted_certificates=TrustAll()
    )
    
    query = """
    MERGE (d:Department {name: $department})
    MERGE (s:Scheme {name: $scheme_name})
    MERGE (b:Beneficiary {group: $beneficiary_group})
    SET s.benefits = $benefit_details
    MERGE (d)-[:OFFERS]->(s)
    MERGE (s)-[:TARGETS]->(b)
    """
    
    with driver.session() as session:
        for item in extracted_schemes:
            session.run(query, 
                        department=item.get("department", "Social Welfare Department"),
                        scheme_name=item.get("scheme_name"),
                        beneficiary_group=item.get("beneficiary_group", "General Public"),
                        benefit_details=item.get("benefit_details", "As per scheme norms"))
            
    driver.close()
    print("Neo4j Knowledge Graph successfully populated!")

if __name__ == "__main__":
    populate_neo4j()