# 🏛️ Tamil Nadu Government Scheme Navigator (GraphRAG)

Meet the **TN Scheme Navigator**—an AI-powered assistant designed to take the guesswork out of finding government welfare programs. 

Instead of hallucinating facts like standard chatbots, it pairs semantic search with a verified knowledge graph to double-check every single detail. Wrapped in a fast, clean web app complete with safety filters and live relationship charts, it makes exploring welfare schemes simple, fast, and completely reliable!

---

## 🏗️ Architectural Blueprint

```mermaid
graph TD
    subgraph Data Ingestion & Storage
        A[TN Govt Portal (tn.gov.in)] -->|Scrape HTML| B(ingest.py)
        A -->|Extract Entities| C(graph_manager.py)
        B -->|Embeddings| D[(ChromaDB Vector Store)]
        C -->|Directed Graph| E[(NetworkX Graph .pkl)]
    end

    subgraph Hybrid RAG & Guardrails
        F[Citizen / User] -->|Query| G{Input Guardrails}
        G -->|Off-Topic| H[Block & Warn]
        G -->|In-Domain| I[Streamlit App (app.py)]
        I -->|Semantic Search| D
        I -->|Graph Traversal| E
        D -->|Vector Context| J[Hybrid RTCFR Prompt]
        E -->|Graph Truth| J
        J -->|GPT-4o Stream| K[Grounded Response & Sub-Graph]
    end

    K --> F

🚀 Key Features
Hybrid GraphRAG Retrieval: Bridges semantic similarity search (ChromaDB) with structural multi-hop knowledge graph traversal (NetworkX) to completely eliminate AI hallucinations.

Input Safety Guardrails: Automatically intercepts and filters out-of-domain prompts (e.g., weather, coding, sports) to keep the chatbot strictly focused on welfare schemes.

Dynamic Sub-Graph Visualization: Instantly renders real-time Matplotlib sub-graphs in the UI for every query to visually verify department-to-beneficiary relationships on the fly.

Automated Graph Artifacts: Automatically exports high-resolution .png and .pdf knowledge graph schemas upon ingestion.

Rigorous Evals: Includes an automated LLM-as-a-Judge evaluation suite (evaluate_rag.py) scoring faithfulness and answer relevance.

🛠️ How It Was Built End-to-End
Web Scraping (ingest.py): Pulled raw welfare data straight from the official tn.gov.in portal using requests and BeautifulSoup.

Chunking & Vector Storage: Chunked text and embedded it via text-embedding-3-small into ChromaDB for lightning-fast semantic retrieval.

Entity Extraction & Knowledge Graph (graph_manager.py): Used GPT-4o with Pydantic schemas to extract entities (Department, Scheme, Beneficiary) and map exact relationships into a NetworkX graph (.pkl).

Automated Graph Export: Generated high-res .png and .pdf visual artifacts on the fly using Matplotlib.

Input Guardrails (app.py): Filtered out-of-domain queries before they reach the core logic.

Hybrid RAG Querying: Combined vector context with absolute graph truth to ground every response.

Response Generation & Visualization: Streamed grounded answers via Streamlit while rendering live sub-graph relationship charts in real time.

📦 Project File Structure
Plaintext
TNGS_RAG_Chatbot/
├── ingest.py               # Scrapes web content and builds ChromaDB vector store
├── graph_manager.py        # Extracts entities/edges with GPT-4o and builds NetworkX graph
├── app.py                  # Main Streamlit web app with chat, guardrails & sub-graph UI
├── evaluate_rag.py         # Automated LLM-as-a-Judge evaluation script
├── requirements.txt        # Python package dependencies
├── .env                    # Environment variables (OpenAI API key)
└── README.md               # Project documentation
⚙️ Installation & Local Setup
1. Clone the Repository
Bash
git clone [https://github.com/your-username/TNGS_RAG_Chatbot.git](https://github.com/your-username/TNGS_RAG_Chatbot.git)
cd TNGS_RAG_Chatbot
2. Create and Activate Virtual Environment
Bash
python3 -m venv myenv
source myenv/bin/activate
3. Install Dependencies
Bash
pip install -r requirements.txt
4. Configure Environment Variables
Create a .env file in the root directory and add your OpenAI API key:

Code snippet
OPENAI_API_KEY=your_actual_openai_api_key_here
5. Build Databases & Run Ingestion
Run these scripts once to scrape the live portal, build your ChromaDB vector store, generate the NetworkX graph, and export visual charts:

Bash
python3 ingest.py
python3 graph_manager.py
6. Launch the Streamlit Web Application
Bash
streamlit run app.py
Open your browser at http://localhost:8501 to chat with the navigator!

📊 Knowledge Graph Stats
Total Nodes Extracted: 64

Total Edges (Relationships): 102

Relationship Types: OFFERS (Department 
→
→ Scheme), TARGETS (Scheme 
→
→ Beneficiary)

🧪 Running Evaluations
To run the automated LLM-as-a-Judge evaluation suite for faithfulness and relevance:

Bash
python3 evaluate_rag.py
