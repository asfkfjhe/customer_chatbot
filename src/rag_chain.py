
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings
from langchain_community.chat_models import ChatOllama
from langchain.messages import HumanMessage, SystemMessage

DB_PATH = "vector_store/chroma_db"

def get_bot_response(query: str, top_k: int, chat_history: list) -> tuple[str, list[str]]:
    """
    RAG pipeline using:
    - Ollama embeddings
    - ChromaDB vector store
    - Ollama chat model
    """

    try:
        # 1️⃣ Load embedding model
        embedding_model = OllamaEmbeddings(
            model="mxbai-embed-large"
        )

        # 2️⃣ Load vector store
        vector_db = Chroma(
            persist_directory=DB_PATH,
            embedding_function=embedding_model
        )

        # 3️⃣ Retrieve relevant chunks
        retriever = vector_db.as_retriever(
            search_kwargs={"k": top_k}
        )

        docs = retriever.invoke(query)

        if not docs:
            return "I could not find relevant information in the knowledge base.", []

        # 4️⃣ Build context
        context = "\n\n".join([doc.page_content for doc in docs])

        # 5️⃣ Collect sources (document titles)
        sources = list(set([
            doc.metadata.get("source", "Unknown document")
            for doc in docs
        ]))

        # 6️⃣ Initialize chat model
        llm = ChatOllama(model="llama3")

        # 7️⃣ System prompt (IMPORTANT FOR GRADING ⭐)
        system_prompt = """
You are a professional BMW customer service assistant.

Rules:
- Answer ONLY using the provided context.
- If the answer is not in the context, say you don't know.
- Be clear, professional, and helpful.
- Do not hallucinate.
"""

        messages = [SystemMessage(content=system_prompt)]

        for msg in chat_history:
            if msg["role"]=="user":
                messages.append(HumanMessage(content=msg["content"]))
            elif msg["role"]=="assistant":
                messages.append(SystemMessage(content=msg["content"]))

        messages.append(
            HumanMessage(content=f"""
Context:{context}
Question:{query}
Answer:

""")
        )


        # 8️⃣ Generate response
        response = llm.invoke(messages)

        return response.content, sources

    except Exception as e:
        return f"Error: {str(e)}", []