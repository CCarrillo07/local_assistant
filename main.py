from assistant import Assistant
from config import CONFIG, MODELS, resolve_model
from llm.ollama_provider import OllamaProvider
from logger import get_logger
from rag.embedder import OllamaEmbedder
from rag.service import RAGService

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are a helpful general-purpose personal assistant.

Help the user with writing, grammar, summarization,
brainstorming, explanations, and everyday questions.

When the user provides document context:
- Answer using only context.
- Do not use general knowledge.
- Preserve facts exactly as stated.
- Do not invent information.
- Follow the document-answering instructions in the user's message.

Be clear, accurate, and concise.
"""

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

    rag_service: RAGService | None = None
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

            if rag_service is None:

                print("\nLoading and embedding documents...")

                rag_service = RAGService(
                    document_directory="documents",
                    embedder=OllamaEmbedder(),
                    chunk_size=100,
                    overlap=20,
                    top_k=CONFIG.rag_top_k,
                    min_score=CONFIG.rag_min_score
                )
                
                rag_service.initialize()

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

        if rag_enabled and rag_service is not None:
            
            model_message= rag_service.buildprompt(
                user_input
            )
            
            if model_message is None:
                
                response = (
                    "The answer could not be found "
                    "in the documents."
                )
                
                print("\nAssistant:")
                print(response)
                
                continue

        print("\nAssistant: ")
        
        try:
            for text in assistant.send_message(user_message=user_input,model_message=model_message):
                print(text, end="", flush=True)
                
            print()

        except Exception as error:

            print(f"\nError: {error}")
        
if __name__ == "__main__":
    main()