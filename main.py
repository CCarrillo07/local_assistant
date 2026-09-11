from assistant import Assistant
from config import CONFIG, MODELS, resolve_model
from llm.ollama_provider import OllamaProvider
from logger import get_logger
from rag.embedder import OllamaEmbedder
from rag.service import RAGService

logger = get_logger(__name__)

# Base instructions used for both general and document-grounded answers.
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
    """Start the assistant and process commands until the user exits."""

    # Read the initial model and RAG behavior from the application config.
    current_model = CONFIG.default_model
    rag_config = CONFIG.rag

    # Only manual mode exposes commands that let the user control RAG.
    rag_controls_available = (
        rag_config.available
        and rag_config.mode == "manual"
        and rag_config.allow_user_control
    )

    # The provider communicates with Ollama, while Assistant manages the
    # conversation history and sends messages to the selected model.
    provider = OllamaProvider(
        model=current_model,
        host=CONFIG.ollama_host
    )

    assistant = Assistant(
        llm=provider,
        system_prompt=SYSTEM_PROMPT
    )

    # The RAG service is created only when the module is available for this
    # deployment. Manual mode leaves it disabled until the user requests it.
    rag_service: RAGService | None = None
    rag_enabled = False

    if rag_config.available:

        rag_service = RAGService(
            document_directory=rag_config.document_directory,
            index_path=rag_config.index_path,
            embedder=OllamaEmbedder(
                model=rag_config.embedding_model
            ),
            chunk_size=rag_config.chunk_size,
            overlap=rag_config.overlap,
            top_k=rag_config.top_k,
            min_score=rag_config.min_score
        )

        # Auto and required modes need the document index immediately because
        # both modes evaluate every regular user question against RAG.
        if rag_config.mode in {"auto", "required"}:

            print(
                "\nLoading and embedding "
                "RAG documents..."
            )

            rag_service.initialize()
            rag_enabled = True

    logger.info(
        "Assistant started with model %s",
        current_model
    )

    # Display only the commands permitted by the active configuration.
    print("\nCommands:")
    print("/models")
    print("/model <name>")
    if rag_controls_available:
        print("/rag on")
        print("/rag off")
        print("/rag status")
    print("/reset")
    print("/exit")

    # Main command and conversation loop.
    while True:

        user_input = input("\nYou: ").strip()

        if not user_input:
            continue

        # Commands handled directly by the application are not sent to the LLM.
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

        # Reject RAG commands when the deployment policy does not allow the
        # user to control the module, such as in auto or required mode.
        rag_commands = {
            "/rag on",
            "/rag off",
            "/rag status"
        }

        if (user_input.lower() in rag_commands and not rag_controls_available):
            print("\nRAG controls are not available "
                  "in this deployment.")
            continue

        # In manual mode, initialize RAG only the first time it is enabled.
        # Later enable operations reuse the existing in-memory vector store.
        if user_input.lower() == "/rag on":

            if rag_service is None:
                print("\nRAG service is not available.")
                continue
            
            if not rag_service.initialized:
                
                print(
                    "\nLoading and embedding "
                    "documents..."
                )
                
                rag_service.initialize()
                
            rag_enabled = True
            assistant.reset()
                
            print("\nRAG mode enabled")
            continue

        # Reset the conversation when switching RAG behavior so earlier
        # non-RAG and RAG messages are not mixed in the same history.
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

        # model_message contains the augmented prompt when relevant document
        # context is found. It stays None for a normal general-assistant call.
        model_message = None

        if rag_enabled and rag_service is not None:

            model_message = rag_service.build_prompt(
                user_input
            )

            if model_message is None:

                # Auto mode falls back to general knowledge. Manual and
                # required modes abstain when the documents do not support an
                # answer, avoiding an ungrounded response.
                if rag_config.mode == "auto":
                    logger.info(
                        "No relevant RAG context found; "
                        "using the general assistant"
                    )

                else:
                    response = (
                        "The answer could not be found "
                        "in the documents."
                    )
                
                    print("\nAssistant:")
                    print(response)

                    continue

        print("\nAssistant: ")

        # Stream response fragments as the local model generates them.
        try:
            for text in assistant.send_message(
                user_message=user_input,
                model_message=model_message
            ):
                print(text, end="", flush=True)

            print()

        # Keep the CLI running if a single LLM request fails.
        except Exception as error:

            print(f"\nError: {error}")
        
if __name__ == "__main__":
    main()
