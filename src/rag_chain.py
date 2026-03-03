from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.messages import HumanMessage, SystemMessage , AIMessage
import time
from functools import wraps

DB_PATH = "vector_store/chroma_db"



def time_it(stage_name):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            start = time.perf_counter()

            result = func(*args, **kwargs)

            end = time.perf_counter()
            print(f"[{stage_name}] took {end - start:.4f} seconds")

            return result
        return wrapper
    return decorator

@time_it("loading embedding models and vector db")
def load_rag_components():
    """Load heavy components ONCE."""

    embedding_model = OllamaEmbeddings(
        model="mxbai-embed-large"
    )

    vector_db = Chroma(
        persist_directory=DB_PATH,
        embedding_function=embedding_model
    )

    llm = ChatOllama(
        model="llama3:8b"
        )

    return vector_db, llm

@time_it("bot response")
def get_bot_response(query, top_k, chat_history, vector_db, llm):

    retriever = vector_db.as_retriever(search_kwargs={"k": top_k})
    docs = retriever.invoke(query)

    if not docs:
        return "I could not find relevant information in the knowledge base.", []

    context = "\n\n".join([doc.page_content for doc in docs])

    sources = list(set([
        doc.metadata.get("source", "Unknown document")
        for doc in docs
    ]))

    system_prompt = """
        You are a professional BMW customer service assistant.

        Decide how to respond.

        If the question is related to:
        - Vehicles
        - Company services
        - Warranty
        - Maintenance
        - Knowledge base documents

        Then answer using the provided context.

        If the question is unrelated to business knowledge,
        respond with:

        "I am sorry, but I can only help with vehicle and service related questions."

        Rules:
        - Do not hallucinate.
        - If unsure, say you don't know.
        - Be professional, helpful, and concise.
        - Greet the user when starting a conversation.
    """

    messages = [SystemMessage(content=system_prompt)]

    for msg in chat_history[-4:]:
        if msg["role"] == "user":
            messages.append(HumanMessage(content=msg["content"]))
        elif msg["role"] == "assistant":
            messages.append(AIMessage(content=msg["content"]))

    messages.append(
        HumanMessage(content=f"""
Context:
{context}

Question:
{query}

Answer:
""")
    )
    
    response = llm.invoke(messages)
    print("answer generated")

    return response.content, sources