from assistant import Assistant
from llm.ollama_provider import OllamaProvider
from config import CONFIG, MODELS, resolve_model
from logger import get_logger

logger = get_logger(__name__)

SYSTEM_PROMPT = """
You are a helpful general-purpose personal assistant.

Help the user with writing, grammar, summarization,
brainstorming, explanations, and everyday questions.

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

    logger.info(
        "Assistant started with model %s",
        current_model
    )
        
    print("\nCommands:")
    print("/models")
    print("/model <name>")
    print("/reset")
    print("exit")
    
    while True:
        
        user_input = input("\nYou: ").strip()
        
        if not user_input:
            continue

        if user_input.lower() == "/exit":
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

        print("\nAssistant: ")
        
        try:
            for text in assistant.send_message(user_input):
                print(text, end="", flush=True)
                
            print()

        except Exception as error:

            print(f"\nError: {error}")
        
if __name__ == "__main__":
    main()