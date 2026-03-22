import os
import faiss
import numpy as np
import pandas as pd
import asyncio
from concurrent.futures import ThreadPoolExecutor
from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from crewai import Agent, Task, Crew, LLM
from functools import lru_cache
import time
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware

# Load variables from .env file
load_dotenv()

# ==========================
# 1. Load Dataset
# ==========================
df = pd.read_csv("support_dataset.csv")

# ==========================
# 2. Embedding Model + FAISS Cache
# ==========================
print("Embedding Model & FAISS Setup...")
model = SentenceTransformer("all-MiniLM-L6-v2")
embedding_file = "embeddings.npy"
index_file = "faiss_index.index"
sources_file = "sources.npy"
print(" Using embedding model:", model)
if os.path.exists(embedding_file) and os.path.exists(index_file) and os.path.exists(sources_file):
    print(" Loading existing embeddings & FAISS index...")
    embeddings = np.load(embedding_file)
    index = faiss.read_index(index_file)
    sources = np.load(sources_file).tolist()
else:
    print(" Creating embeddings & FAISS index for the first time...")
    combined_texts = df["question"].tolist() + df["answer"].tolist()
    embeddings = model.encode(combined_texts, convert_to_tensor=False)
    np.save(embedding_file, embeddings)

    dimension = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings))
    faiss.write_index(index, index_file)

    sources = ["Q"] * len(df) + ["A"] * len(df)
    np.save(sources_file, sources)
print("Embeddings loaded successfully")

# ==========================
# Thread Pool for CPU-bound operations
# ==========================
executor = ThreadPoolExecutor(max_workers=4)

# ==========================
# Async Retrieval
# ==========================
def faiss_lookup(query, top_k=2):
    """Synchronous FAISS lookup"""
    query_embedding = model.encode([query], convert_to_tensor=False)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    distances, indices = index.search(query_embedding, top_k)

    results = []
    for idx, dist in zip(indices[0], distances[0]):
        if idx == -1:
            continue
        if sources[idx] == "Q":
            ans = df.iloc[idx]["answer"]
            results.append((ans, dist))
        else:
            ans_idx = idx - len(df)
            ans = df.iloc[ans_idx]["answer"]
            results.append((ans, dist))

    # Remove duplicates while preserving order
    seen = set()
    unique_answers = []
    for answer, dist in results:
        if answer not in seen:
            unique_answers.append(answer)
            seen.add(answer)

    if unique_answers:
        return "\n".join(unique_answers)
    return "NO_MATCH"

# ==========================
# 3. LLM Setup
# ==========================
print("LLM Setup...")
huggingface_key = os.getenv("HUGGINGFACE_API_KEY")
llm = LLM(
    model="huggingface/deepseek-ai/DeepSeek-V3.1-Terminus",
    api_key=huggingface_key,
)
print("LLM Setup successfully")

# ==========================
# 4. Agents
# ==========================
support_agent = Agent(
    role="Customer Support Agent",
    goal="Provide accurate answers using only the retrieved knowledge base information.",
    backstory=(
        "You are a restaurant customer support agent. Answer ONLY using retrieved_answer. "
        "If it's 'NO_MATCH', respond with exactly 'NO_MATCH'."
    ),
    allow_delegation=False,
    verbose=False,
    llm=llm
)

review_agent = Agent(
    role="Response Quality Controller",
    goal="Deliver final customer response.",
    backstory=(
        "If support agent returned 'NO_MATCH':\n"
        "- If query is restaurant-related → 'Please contact us at restaurant@xyz.com'\n"
        "- Else → 'Your message does not seem related to our restaurant.'"
    ),
    allow_delegation=False,
    verbose=False,
    llm=llm
)

# ==========================
# 5. Tasks
# ==========================
inquiry_task = Task(
    description=(
        "Customer Query: {customer_query}\n"
        "Retrieved Information: {retrieved_answer}\n\n"
        "Conversation History:\n{chat_history}\n\n"
        "Instructions:\n"
        "- If retrieved_answer is 'NO_MATCH', respond with exactly 'NO_MATCH'\n"
        "- Otherwise, answer ONLY using retrieved_answer, considering chat history"
    ),
    agent=support_agent,
    expected_output="Either 'NO_MATCH' or a concise answer."
)

qa_task = Task(
description=(
    "The customer asked: {customer_query}\n\n"
    "Conversation so far:\n{chat_history}\n\n"
    "If support agent responded with anything other than 'NO_MATCH', return it directly.\n"
    "If it was 'NO_MATCH':\n"
    "  • If greeting (hello, hi, hey, good morning, good evening, etc.) → "
    "'Hello! How can I help you?'\n"
    "  • If gratitude/thanks (thanks, thank you, thx, much appreciated, etc.) → "
    "'You're welcome! Good to talk with you. If you need further help, feel free to ask me.'\n"
    "  • If customer explicitly asks for help, mentions an issue, problem, or says phrases like "
    "'help', 'issue', 'problem', 'want to ask', 'want to know' → "
    "'Sure! How can I help you?'\n"
    "  • If farewell/bye (bye, goodbye, see you, take care, good night, etc.) → "
    "'Goodbye! Have a nice day.'\n"
    "  • If someone asks about the restaurant name → 'Our restaurant is called AN Cafe.'\n"
    "  • If restaurant-related → 'Please contact us at restaurant@xyz.com'\n"
    "  • Otherwise → 'Your message does not seem related to our restaurant.'"
),
    agent=review_agent,
    expected_output="Final customer response.",
    context=[inquiry_task]
)

crew = Crew(
    agents=[support_agent, review_agent],
    tasks=[inquiry_task, qa_task],
    verbose=False
)

# ==========================
# 6. FastAPI App
# ==========================
app = FastAPI(title="Customer Support API", version="1.1")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    query: str
    session_id: str = "default"   # optional session id

# ==========================
# 7. Conversation Memory
# ==========================
conversation_memory = {}  # { session_id: [("user", msg), ("bot", msg), ...] }

def build_chat_history(session_id: str) -> str:
    """Convert memory to string"""
    history = conversation_memory.get(session_id, [])
    return "\n".join([f"{role}: {msg}" for role, msg in history])

async def get_answer_async(query: str, session_id: str):
    """Async query processor with memory"""
    loop = asyncio.get_event_loop()
    retrieved_context = faiss_lookup(query)

    chat_history = build_chat_history(session_id)

    inputs = {
        "customer_query": query,
        "retrieved_answer": retrieved_context,
        "chat_history": chat_history
    }
    result = await loop.run_in_executor(executor, crew.kickoff, inputs)

    try:
        final_answer = result["raw"] if isinstance(result, dict) else result.raw
    except:
        final_answer = str(result)

    # Update memory
    if session_id not in conversation_memory:
        conversation_memory[session_id] = []
    conversation_memory[session_id].append(("user", query))
    conversation_memory[session_id].append(("bot", final_answer))

    return final_answer, retrieved_context

# ==========================
# 8. Endpoints
# ==========================
@app.post("/ask")
async def ask_question(request: QueryRequest):
    try:
        final_answer, retrieved_context = await get_answer_async(request.query, request.session_id)
        return {
            "query": request.query,
            "session_id": request.session_id,
            "retrieved_context": retrieved_context,
            "answer": final_answer,
            "chat_history": conversation_memory.get(request.session_id, [])
        }
    except Exception as e:
        return {
            "query": request.query,
            "session_id": request.session_id,
            "retrieved_context": "",
            "answer": f"Error processing query: {str(e)}",
            "error": True
        }

@app.get("/health")
async def health_check():
    return {"status": "healthy", "timestamp": time.time()}


#uvicorn customer_support:app --host 0.0.0.0 --port 8002
    
