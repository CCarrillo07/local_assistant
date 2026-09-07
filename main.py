from assistant import Assistant
from config import CONFIG, MODELS, resolve_model
from llm.ollama_provider import OllamaProvider
from logger import get_logger
from rag.chunker import chunk_documents
from rag.embedder import OllamaEmbedder
from rag.loader import load_documents
from rag.prompt import build_rag_prompt
from rag.vector_store import InMemoryVectorStore

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are a helpful general-purpose personal assistant.

Help the user with writing, grammar, summarization,
brainstorming, explanations, and everyday questions.

Be clear, accurate, and concise.
"""

def build_vector_store() -> InMemoryVectorStore:

    documents = load_documents("documents")

    chunks = chunk_documents(
        documents,
        chunk_size=100,
        overlap=20
    )

    embedder = OllamaEmbedder()

    vector_store = InMemoryVectorStore(embedder)

    vector_store.add_chunks(chunks)

    return vector_store

def main():
    
    current_model = CONFIG.default_model
    
    provider = OllamaProvider(
        model=current_model,
        host=CONFIG.ollama_host
    )

    
    assistant = Assistant(
        llm=provider,
        system_prompt=SYSTEM_PROMPT
    )

    vector_store: InMemoryVectorStore | None = None
    rag_enabled = False

    logger.info(
        "Assistant started with model %s",
        current_model
    )
        
    print("\nCommands:")
    print("/models")
    print("/model <name>")
    print("/rag on")
    print("/rag off")
    print("/rag status")
    print("/reset")
    print("/exit")
    
    while True:
        
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue

        if user_input.lower() in {"exit", "/exit"}:
            break

        if user_input.lower() == "/reset":
            assistant.reset()
            continue
        
        if user_input.lower() == "/models":
            
            print("\nAvailable models:")
            
            for alias, model in MODELS.items():
                print(f"{alias}: {model}")
                
            continue
        
        if user_input.lower().startswith("/model "):
            
            requested_model = user_input.split(
                maxsplit=1
            )[1]
            
            current_model = resolve_model(
                requested_model
            )
            
            assistant.change_model(
                current_model
            )
            
            continue

        if user_input.lower() == "/rag on":

            if vector_store is None:

                print("\nLoading and embedding documents...")

                vector_store = build_vector_store()

            rag_enabled = True
            assistant.reset()

            print("\nRAG mode enabled")
            continue

        if user_input.lower() == "/rag off":

            rag_enabled = False
            assistant.reset()

            print("\nRAG mode disabled")
            continue

        if user_input.lower() == "/rag status":

            status = (
                "enabled"
                if rag_enabled
                else "disabled"
            )

            print(f"RAG mode is {status}")
            continue

        model_message = None

        if rag_enabled and vector_store is not None:
            
            results = vector_store.search(
                query=user_input,
                top_k=CONFIG.rag_top_k,
                min_score=CONFIG.rag_min_score
            )

            if not results:

                response = (
                    "The answer could not be found "
                    "in the documents."
                )

                print("\nAssistant:")
                print(response)

                continue

            model_message= build_rag_prompt(
                question=user_input,
                results=results
            )


        print("\nAssistant: ")
        
        try:
            for text in assistant.send_message(user_message=user_input,model_message=model_message):
                print(text, end="", flush=True)
                
            print()

        except Exception as error:

            print(f"\nError: {error}")
        
if __name__ == "__main__":
    main()