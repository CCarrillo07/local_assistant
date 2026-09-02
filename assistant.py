from llm.base import LLMProvider, Message

class Assistant:
    
    def __init__(
        self,
        llm: LLMProvider,
        system_prompt: str
    ):
        self.llm = llm
        self.system_prompt = system_prompt
        
        self.messages: list[Message] = [
            {
                "role": "sytem",
                "content": self.system_prompt
            }
        ]
    
    def send_message(self, user_message: str):
        
        self.messages.append(
            {
                "role": "user",
                "content": user_message
            }
        )
        
        response = ""
        
        for text in self.llm.stream_chat(self.messages):
            response += text
            yield text

        self.messages.append(
            {
                "role": "assistant",
                "content": response
            }
        )
    
    def change_model(self, model: str) -> None:
        self.llm.set_model(model)
        self.reset()
    
    def reset(self):
        
        self.messages = [
            {
                "role": "system",
                "content": self.system_prompt
            }
        ]
    
    