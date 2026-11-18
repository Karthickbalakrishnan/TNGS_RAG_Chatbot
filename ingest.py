import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Target URL for the Tamil Nadu Government Scheme List
URL = "https://www.tn.gov.in/scheme_list.php?dep_id=Mg=="

def scrape_schemes():
    print("Fetching data from TN Government portal...")
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    }
    response = requests.get(URL, headers=headers)
    if response.status_code != 200:
        raise Exception(f"Failed to fetch webpage. Status code: {response.status_code}")
        
    soup = BeautifulSoup(response.text, 'html.parser')
    
    # Extract scheme rows, titles, and text blocks
    schemes_data = []
    for item in soup.find_all(['tr', 'div', 'p'], class_=['row', 'scheme-item', 'content']):
        text = item.get_text(separator=" ", strip=True)
        if len(text) > 40:  # Filter out small navigation elements/noise
            schemes_data.append(text)
            
    # Fallback if specific classes aren't matched: get all text from body
    if not schemes_data:
        body = soup.find('body')
        schemes_data = [body.get_text(separator=" ", strip=True)]
        
    raw_text = "\n\n".join(schemes_data)
    print(f"Successfully scraped {len(raw_text)} characters of text.")
    return raw_text

def build_vector_store():
    raw_text = scrape_schemes()
    
    # Chunking text into semantic pieces for vector retrieval
    print("Splitting text into chunks...")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500, 
        chunk_overlap=50,
        separators=["\n\n", "\n", " ", ""]
    )
    docs = text_splitter.create_documents([raw_text])
    print(f"Created {len(docs)} text chunks.")
    
    # Generate embeddings and store in ChromaDB
    print("Generating embeddings and saving to ChromaDB...")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents=docs, 
        embedding=embeddings, 
        persist_directory="./chroma_db"
    )
    print("Vector database successfully built and persisted at './chroma_db'!")

if __name__ == "__main__":
    build_vector_store()