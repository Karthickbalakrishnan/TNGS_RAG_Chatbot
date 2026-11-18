import os
import re
from dotenv import load_dotenv
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage

# Load environment variables from .env file
load_dotenv()

# Define the conversational RTCFR-structured prompt template with memory support
rtcfr_chat_prompt = ChatPromptTemplate.from_messages([
    ("system", """
[ROLE]
You are a warm, empathetic, and expert AI Government Scheme Navigator for the Government of Tamil Nadu. Your job is to help citizens accurately discover welfare schemes, eligibility rules, and application procedures based exclusively on the provided context, while maintaining a polite, supportive, and conversational tone.

[CONTEXT]
{context}

[TASK]
Answer the user's question accurately using ONLY the retrieved context and conversation history. 
- If the user makes small talk, respond warmly, address them by their name ({user_name}), and guide them back to finding welfare schemes.
- If information is not present in the context, politely state that you cannot find the record in the current Tamil Nadu government database.
- Keep track of the user's name and previous questions.

[RESPONSE FORMAT]
- Use clear headings and bullet points for eligibility, benefits, and steps.
- Conclude your answer by asking if they would like to know about a specific eligibility criteria or application mode.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{question}")
])

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def extract_name(user_input):
    # Look for patterns like "I'm Karthick", "I am Karthick", "name is Karthick"
    match = re.search(r"(?:i'm|i am|name is|this is)\s+([a-zA-Z]+)", user_input, re.IGNORECASE)
    if match:
        return match.group(1).capitalize()
    
    # Filter out common greeting words to find the proper name
    common_words = {"hi", "hello", "im", "i'm", "doing", "well", "thanks", "thank", "you", "good", "great", "hey", "and"}
    words = re.findall(r"[a-zA-Z]+", user_input)
    
    for word in words:
        if word.lower() not in common_words and len(word) > 1:
            return word.capitalize()
            
    return "Citizen"

def run_enhanced_chatbot():
    # 1. Load embeddings and existing Chroma vector store
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    if not os.path.exists("./chroma_db"):
        print("Error: './chroma_db' not found. Please run 'ingest.py' first!")
        return

    vector_store = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    
    # 2. Initialize LLM with streaming enabled
    llm = ChatOpenAI(model="gpt-4o", temperature=0.3, streaming=True)
    
    print("\n" + "="*60)
    print("🤖 Welcome to the TN Government Scheme Navigator!")
    print("="*60)
    
    # 3. Clean Greeting & Personalization
    raw_intro = input("Hello! How is your day going? What is your name?\n> ")
    user_name = extract_name(raw_intro)
    
    print(f"\nIt's wonderful to meet you, {user_name}! I hope your day is going great.")
    print(f"I'm ready to help you explore Tamil Nadu government welfare programs. Type 'exit' anytime to quit.\n" + "-"*50)
    
    # Initialize chat history memory list
    chat_history = []
    
    while True:
        query = input(f"\n{user_name}: ")
        if query.lower() == 'exit':
            print(f"\nGoodbye, {user_name}! Have a wonderful day ahead!")
            break
        if not query.strip():
            continue
            
        print("\nAssistant: ", end="", flush=True)
        
        try:
            # Retrieve relevant documents based on query
            docs = retriever.invoke(query)
            context_text = format_docs(docs)
            
            # Format prompt with context, history, user name, and question
            prompt_formatted = rtcfr_chat_prompt.invoke({
                "context": context_text,
                "chat_history": chat_history,
                "user_name": user_name,
                "question": query
            })
            
            # Stream the response chunk by chunk in real-time
            full_response = ""
            for chunk in llm.stream(prompt_formatted):
                content = chunk.content
                print(content, end="", flush=True)
                full_response += content
                
            print("\n" + "-"*50)
            
            # Update conversational memory
            chat_history.append(HumanMessage(content=query))
            chat_history.append(AIMessage(content=full_response))
            
        except Exception as e:
            print(f"\nAn error occurred: {e}")

if __name__ == "__main__":
    run_enhanced_chatbot()