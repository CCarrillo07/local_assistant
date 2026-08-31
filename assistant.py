from ollama import chat

MODEL = "qwen3.5:2b"

SYSTEM_PROMPT = """
You are a helpful general-purpose assistant.

Help the user with writing, grammar, summarization,
brainstorming, explanations, and every day questions.

Be clear, accurate, and concise.
"""

def main():
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT   
        }
        
    ]

    print(f"Local Assistant - {MODEL}. \n")
    print("Type /exit to quit.\n")

    while True:
        user_input = input("\nYou: ").strip()

        if user_input.lower() == "/exit":
            break

        if not user_input:
            continue

        messages.append(
            {
                "role": "user",
                "content": user_input
            }
        )

        print("\nAsssistant: ", end="", flush=True)

        response_text = ""

        stream = chat(
            model=MODEL,
            messages=messages,
            stream=True
        )

        for chunk in stream:
            text = chunk["message"]["content"]

            print(text, end="", flush=True)

            response_text += text

        print()

        messages.append(
            {
                "role": "asssistant",
                "content": response_text
            }
        )

if __name__ == "__main__":
    main()